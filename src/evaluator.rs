//! Lock-free CPU-to-GPU evaluation transport.

use std::cell::UnsafeCell;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, unbounded, Receiver, RecvTimeoutError, Sender, TryRecvError};

use crate::encoding::{EncodedBoard, PolicyMask, INPUT_SIZE, POLICY_SIZE};

const SLOT_FREE: u8 = 0;
const SLOT_PENDING: u8 = 1;
const SLOT_RUNNING: u8 = 2;
const SLOT_DONE: u8 = 3;
const SLOT_ERROR: u8 = 4;
const SLOT_CANCELLED: u8 = 5;
const STOP_REQUEST: usize = usize::MAX;
const SUBMIT_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_WAIT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone)]
pub enum EvaluationError {
    Closed,
    Busy,
    Timeout,
    Backend(String),
    Protocol,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("GPU evaluator is closed"),
            Self::Busy => f.write_str("GPU evaluator has no free evaluation slot"),
            Self::Timeout => f.write_str("GPU evaluator operation timed out"),
            Self::Backend(message) => write!(f, "GPU evaluator backend failed: {message}"),
            Self::Protocol => f.write_str("evaluation slot protocol violation"),
        }
    }
}

impl std::error::Error for EvaluationError {}

#[derive(Clone)]
pub struct Evaluation {
    pub policy: [f32; POLICY_SIZE],
    pub value: f32,
    pub wdl: [f32; 3],
}

impl Default for Evaluation {
    fn default() -> Self {
        Self {
            policy: [0.0; POLICY_SIZE],
            value: 0.0,
            wdl: [0.0; 3],
        }
    }
}

struct SlotData {
    input: EncodedBoard,
    legal: PolicyMask,
    output: Evaluation,
    error: Option<String>,
}

impl Default for SlotData {
    fn default() -> Self {
        Self {
            input: [0.0; INPUT_SIZE],
            legal: PolicyMask::default(),
            output: Evaluation::default(),
            error: None,
        }
    }
}

struct EvalSlot {
    state: AtomicU8,
    data: UnsafeCell<SlotData>,
}

unsafe impl Send for EvalSlot {}
unsafe impl Sync for EvalSlot {}

impl EvalSlot {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(SLOT_FREE),
            data: UnsafeCell::new(SlotData::default()),
        }
    }
}

trait BatchBackend: Send + 'static {
    fn evaluate_slots(&mut self, slots: &[(usize, &EvalSlot)]) -> Result<(), EvaluationError>;
}

pub struct UniformBackend;

impl BatchBackend for UniformBackend {
    fn evaluate_slots(&mut self, slots: &[(usize, &EvalSlot)]) -> Result<(), EvaluationError> {
        for (_, slot) in slots {
            let data = unsafe { &mut *slot.data.get() };
            let legal_count = data
                .legal
                .0
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>();
            data.output.policy.fill(0.0);
            if legal_count != 0 {
                let probability = 1.0 / legal_count as f32;
                for (word_index, word) in data.legal.0.iter().copied().enumerate() {
                    let mut remaining = word;
                    while remaining != 0 {
                        let bit = remaining.trailing_zeros() as usize;
                        let index = word_index * u64::BITS as usize + bit;
                        if index < POLICY_SIZE {
                            data.output.policy[index] = probability;
                        }
                        remaining &= remaining - 1;
                    }
                }
            }
            data.output.value = 0.0;
            data.output.wdl = [0.0, 1.0, 0.0];
        }
        Ok(())
    }
}

pub struct SharedGpuEvaluator {
    slots: Arc<Vec<EvalSlot>>,
    free_tx: Sender<usize>,
    free_rx: Receiver<usize>,
    request_tx: Sender<usize>,
    closed: AtomicBool,
    actor_finished: AtomicBool,
    actor: Mutex<Option<JoinHandle<()>>>,
}

pub struct EvaluationTicket {
    evaluator: Arc<SharedGpuEvaluator>,
    slot_index: Option<usize>,
}

impl SharedGpuEvaluator {
    pub fn uniform(slot_count: usize, max_batch_size: usize, max_wait: Duration) -> Arc<Self> {
        Self::start(UniformBackend, slot_count, max_batch_size, max_wait)
    }

    fn start<B: BatchBackend>(
        backend: B,
        slot_count: usize,
        max_batch_size: usize,
        max_wait: Duration,
    ) -> Arc<Self> {
        let slot_count = slot_count.max(1);
        let max_batch_size = max_batch_size.clamp(1, slot_count);
        let (free_tx, free_rx) = bounded(slot_count);
        // At most `slot_count` requests can be active, so an unbounded channel
        // cannot grow with useful work. It does guarantee that shutdown can
        // enqueue the sentinel even when every slot is currently pending.
        let (request_tx, request_rx) = unbounded();
        let mut slots = Vec::with_capacity(slot_count);
        for index in 0..slot_count {
            slots.push(EvalSlot::new());
            let _ = free_tx.send(index);
        }
        let evaluator = Arc::new(Self {
            slots: Arc::new(slots),
            free_tx,
            free_rx,
            request_tx,
            closed: AtomicBool::new(false),
            actor_finished: AtomicBool::new(false),
            actor: Mutex::new(None),
        });

        let actor_owner = Arc::downgrade(&evaluator);
        let actor_slots = Arc::clone(&evaluator.slots);
        let handle = thread::Builder::new()
            .name("zero-gpu-batch-evaluator".to_owned())
            .spawn(move || {
                run_actor(
                    actor_owner,
                    actor_slots,
                    request_rx,
                    backend,
                    max_batch_size,
                    max_wait,
                )
            });
        if let Ok(handle) = handle {
            if let Ok(mut actor) = evaluator.actor.lock() {
                *actor = Some(handle);
            }
        } else {
            evaluator.closed.store(true, Ordering::Release);
            evaluator.actor_finished.store(true, Ordering::Release);
        }
        evaluator
    }

    pub fn slot_capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn try_submit(
        self: &Arc<Self>,
        input: &EncodedBoard,
        legal: PolicyMask,
    ) -> Result<EvaluationTicket, EvaluationError> {
        if self.closed.load(Ordering::Acquire) || self.actor_finished.load(Ordering::Acquire) {
            return Err(EvaluationError::Closed);
        }
        let slot_index = match self.free_rx.try_recv() {
            Ok(index) => index,
            Err(TryRecvError::Empty) => return Err(EvaluationError::Busy),
            Err(TryRecvError::Disconnected) => return Err(EvaluationError::Closed),
        };
        let Some(slot) = self.slots.get(slot_index) else {
            let _ = self.free_tx.send(slot_index);
            return Err(EvaluationError::Protocol);
        };
        if slot.state.load(Ordering::Acquire) != SLOT_FREE {
            let _ = self.free_tx.send(slot_index);
            return Err(EvaluationError::Protocol);
        }
        unsafe {
            let data = &mut *slot.data.get();
            data.input.copy_from_slice(input);
            data.legal = legal;
            data.error = None;
        }
        slot.state.store(SLOT_PENDING, Ordering::Release);
        if self.request_tx.send(slot_index).is_err() {
            slot.state.store(SLOT_FREE, Ordering::Release);
            let _ = self.free_tx.send(slot_index);
            return Err(EvaluationError::Closed);
        }

        Ok(EvaluationTicket {
            evaluator: Arc::clone(self),
            slot_index: Some(slot_index),
        })
    }

    pub fn submit(
        self: &Arc<Self>,
        input: &EncodedBoard,
        legal: PolicyMask,
    ) -> Result<EvaluationTicket, EvaluationError> {
        let deadline = Instant::now() + SUBMIT_TIMEOUT;
        loop {
            match self.try_submit(input, legal) {
                Ok(ticket) => return Ok(ticket),
                Err(EvaluationError::Busy) => {
                    if Instant::now() >= deadline {
                        return Err(EvaluationError::Timeout);
                    }
                    thread::sleep(Duration::from_micros(50));
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub fn evaluate(
        self: &Arc<Self>,
        input: &EncodedBoard,
        legal: PolicyMask,
    ) -> Result<Evaluation, EvaluationError> {
        self.submit(input, legal)?.wait()
    }

    fn wait_slot(&self, slot_index: usize) -> Result<Evaluation, EvaluationError> {
        self.wait_slot_until(slot_index, Instant::now() + SUBMIT_TIMEOUT)
    }

    fn wait_slot_until(
        &self,
        slot_index: usize,
        deadline: Instant,
    ) -> Result<Evaluation, EvaluationError> {
        let slot = self
            .slots
            .get(slot_index)
            .ok_or(EvaluationError::Protocol)?;

        let mut spins = 0_u32;
        loop {
            match slot.state.load(Ordering::Acquire) {
                SLOT_DONE => {
                    let result = unsafe { (&*slot.data.get()).output.clone() };
                    if slot
                        .state
                        .compare_exchange(SLOT_DONE, SLOT_FREE, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let _ = self.free_tx.send(slot_index);
                        return Ok(result);
                    }
                }
                SLOT_ERROR => {
                    let message = unsafe { (&*slot.data.get()).error.clone() }
                        .unwrap_or_else(|| "actor evaluation failed".to_owned());
                    if slot
                        .state
                        .compare_exchange(
                            SLOT_ERROR,
                            SLOT_FREE,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        let _ = self.free_tx.send(slot_index);
                        return Err(EvaluationError::Backend(message));
                    }
                }
                SLOT_PENDING | SLOT_RUNNING => {
                    if self.actor_finished.load(Ordering::Acquire) {
                        let released = slot
                            .state
                            .compare_exchange(
                                SLOT_PENDING,
                                SLOT_FREE,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                            || slot
                                .state
                                .compare_exchange(
                                    SLOT_RUNNING,
                                    SLOT_FREE,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_ok();
                        if released {
                            let _ = self.free_tx.send(slot_index);
                        }
                        return Err(EvaluationError::Closed);
                    }
                    if Instant::now() >= deadline {
                        if slot
                            .state
                            .compare_exchange(
                                SLOT_PENDING,
                                SLOT_CANCELLED,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                            || slot
                                .state
                                .compare_exchange(
                                    SLOT_RUNNING,
                                    SLOT_CANCELLED,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                )
                                .is_ok()
                        {
                            return Err(EvaluationError::Timeout);
                        }
                        continue;
                    }
                    if spins < 128 {
                        std::hint::spin_loop();
                    } else {
                        thread::sleep(Duration::from_micros(50));
                    }
                    spins = spins.saturating_add(1);
                }
                SLOT_FREE => return Err(EvaluationError::Protocol),
                SLOT_CANCELLED => return Err(EvaluationError::Timeout),
                _ => return Err(EvaluationError::Closed),
            }
        }
    }

    pub fn shutdown(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            let _ = self.request_tx.send(STOP_REQUEST);
        }
        let handle = self.actor.lock().ok().and_then(|mut actor| actor.take());
        if let Some(handle) = handle {
            finish_actor(handle, &self.actor_finished);
        }
    }

    pub fn is_healthy(&self) -> bool {
        !self.closed.load(Ordering::Acquire) && !self.actor_finished.load(Ordering::Acquire)
    }
}

impl EvaluationTicket {
    pub fn wait(mut self) -> Result<Evaluation, EvaluationError> {
        let index = self.slot_index.take().ok_or(EvaluationError::Protocol)?;
        self.evaluator.wait_slot(index)
    }

    pub fn wait_timeout(mut self, timeout: Duration) -> Result<Evaluation, EvaluationError> {
        let index = self.slot_index.take().ok_or(EvaluationError::Protocol)?;
        self.evaluator
            .wait_slot_until(index, Instant::now() + timeout)
    }
}

impl Drop for EvaluationTicket {
    fn drop(&mut self) {
        if let Some(index) = self.slot_index.take() {
            let _ = self.evaluator.wait_slot(index);
        }
    }
}

impl Drop for SharedGpuEvaluator {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        let _ = self.request_tx.send(STOP_REQUEST);
        if let Ok(actor) = self.actor.get_mut() {
            if let Some(handle) = actor.take() {
                finish_actor(handle, &self.actor_finished);
            }
        }
    }
}

fn finish_actor(handle: JoinHandle<()>, actor_finished: &AtomicBool) {
    let deadline = Instant::now() + SHUTDOWN_WAIT;
    while !actor_finished.load(Ordering::Acquire) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    if actor_finished.load(Ordering::Acquire) {
        let _ = handle.join();
    }
    // A stuck foreign backend cannot be forcefully cancelled from Rust. Drop
    // the handle after the bounded wait so shutdown does not deadlock the host.
}

fn run_actor<B: BatchBackend>(
    evaluator: Weak<SharedGpuEvaluator>,
    slots: Arc<Vec<EvalSlot>>,
    requests: Receiver<usize>,
    mut backend: B,
    max_batch_size: usize,
    max_wait: Duration,
) {
    struct ActorExit {
        evaluator: Weak<SharedGpuEvaluator>,
        slots: Arc<Vec<EvalSlot>>,
    }
    impl Drop for ActorExit {
        fn drop(&mut self) {
            if let Some(evaluator) = self.evaluator.upgrade() {
                evaluator.closed.store(true, Ordering::Release);
                for (index, slot) in self.slots.iter().enumerate() {
                    let state = slot.state.load(Ordering::Acquire);
                    let mut failed = false;
                    if state == SLOT_PENDING || state == SLOT_RUNNING {
                        unsafe {
                            (*slot.data.get()).error =
                                Some("evaluator actor terminated unexpectedly".to_owned());
                        }
                        failed = slot
                            .state
                            .compare_exchange(
                                state,
                                SLOT_ERROR,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok();
                    }
                    if !failed
                        && slot
                            .state
                            .compare_exchange(
                                SLOT_CANCELLED,
                                SLOT_FREE,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                    {
                        let _ = evaluator.free_tx.send(index);
                    }
                }
                evaluator.actor_finished.store(true, Ordering::Release);
            }
        }
    }

    let _actor_exit = ActorExit {
        evaluator: Weak::clone(&evaluator),
        slots: Arc::clone(&slots),
    };
    let mut ids = Vec::with_capacity(max_batch_size);
    let mut batch = Vec::with_capacity(max_batch_size);
    loop {
        let first = match requests.recv() {
            Ok(STOP_REQUEST) | Err(_) => {
                if let Some(shared) = evaluator.upgrade() {
                    fail_pending(&shared, &requests);
                }
                break;
            }
            Ok(index) => index,
        };
        let Some(shared) = evaluator.upgrade() else {
            break;
        };
        ids.clear();
        batch.clear();
        ids.push(first);
        let mut shutdown_requested = false;
        let deadline = Instant::now() + max_wait;
        while ids.len() < max_batch_size {
            match requests.try_recv() {
                Ok(STOP_REQUEST) => {
                    shared.closed.store(true, Ordering::Release);
                    shutdown_requested = true;
                    break;
                }
                Ok(index) => ids.push(index),
                Err(TryRecvError::Empty) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match requests.recv_timeout(remaining) {
                        Ok(STOP_REQUEST) => {
                            shared.closed.store(true, Ordering::Release);
                            shutdown_requested = true;
                            break;
                        }
                        Ok(index) => ids.push(index),
                        Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                    }
                }
                Err(TryRecvError::Disconnected) => break,
            }
        }

        for &index in &ids {
            let Some(slot) = slots.get(index) else {
                continue;
            };
            if slot
                .state
                .compare_exchange(
                    SLOT_PENDING,
                    SLOT_RUNNING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                batch.push((index, slot));
            } else if slot
                .state
                .compare_exchange(
                    SLOT_CANCELLED,
                    SLOT_FREE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let _ = shared.free_tx.send(index);
            }
        }
        if batch.is_empty() {
            if shutdown_requested {
                fail_pending(&shared, &requests);
                break;
            }
            continue;
        }
        let backend_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.evaluate_slots(&batch)
        }));
        let (success, error_message) = match backend_result {
            Ok(Ok(())) => (true, None),
            Ok(Err(error)) => (
                false,
                Some(match error {
                    EvaluationError::Backend(message) => message,
                    other => other.to_string(),
                }),
            ),
            Err(_) => (false, Some("backend evaluation panicked".to_owned())),
        };
        for (index, slot) in &batch {
            let target = if success { SLOT_DONE } else { SLOT_ERROR };
            if let Some(message) = &error_message {
                unsafe {
                    (*slot.data.get()).error = Some(message.clone());
                }
            }
            if slot
                .state
                .compare_exchange(SLOT_RUNNING, target, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                if slot
                    .state
                    .compare_exchange(
                        SLOT_CANCELLED,
                        SLOT_FREE,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    let _ = shared.free_tx.send(*index);
                }
                continue;
            }
        }
        let panicked = error_message
            .as_deref()
            .is_some_and(|message| message == "backend evaluation panicked");
        if shutdown_requested || panicked || !success {
            shared.closed.store(true, Ordering::Release);
            fail_pending(&shared, &requests);
            break;
        }
    }
}

fn fail_pending(evaluator: &SharedGpuEvaluator, requests: &Receiver<usize>) {
    while let Ok(index) = requests.try_recv() {
        if index == STOP_REQUEST {
            continue;
        }
        if let Some(slot) = evaluator.slots.get(index) {
            if slot.state.load(Ordering::Acquire) == SLOT_PENDING {
                unsafe {
                    (*slot.data.get()).error = Some("evaluator closed".to_owned());
                }
                let marked = slot
                    .state
                    .compare_exchange(
                        SLOT_PENDING,
                        SLOT_ERROR,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok();
                if !marked
                    && slot
                        .state
                        .compare_exchange(
                            SLOT_CANCELLED,
                            SLOT_FREE,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                {
                    let _ = evaluator.free_tx.send(index);
                }
            } else if slot
                .state
                .compare_exchange(
                    SLOT_CANCELLED,
                    SLOT_FREE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                let _ = evaluator.free_tx.send(index);
            }
        }
    }
    for (index, slot) in evaluator.slots.iter().enumerate() {
        if slot.state.load(Ordering::Acquire) == SLOT_PENDING {
            unsafe {
                (*slot.data.get()).error = Some("evaluator closed".to_owned());
            }
            let marked = slot
                .state
                .compare_exchange(
                    SLOT_PENDING,
                    SLOT_ERROR,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
            if !marked
                && slot
                    .state
                    .compare_exchange(
                        SLOT_CANCELLED,
                        SLOT_FREE,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
            {
                let _ = evaluator.free_tx.send(index);
            }
        } else if slot
            .state
            .compare_exchange(
                SLOT_CANCELLED,
                SLOT_FREE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            let _ = evaluator.free_tx.send(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanickingBackend;

    impl BatchBackend for PanickingBackend {
        fn evaluate_slots(&mut self, _slots: &[(usize, &EvalSlot)]) -> Result<(), EvaluationError> {
            panic!("synthetic backend panic")
        }
    }

    struct FailingBackend;

    impl BatchBackend for FailingBackend {
        fn evaluate_slots(&mut self, _slots: &[(usize, &EvalSlot)]) -> Result<(), EvaluationError> {
            Err(EvaluationError::Backend(
                "preserved backend detail".to_owned(),
            ))
        }
    }

    #[test]
    fn backend_panics_are_reported_to_waiting_tickets() {
        let evaluator = SharedGpuEvaluator::start(PanickingBackend, 1, 1, Duration::ZERO);
        let input = [0.0; INPUT_SIZE];
        let ticket = evaluator
            .submit(&input, PolicyMask::default())
            .expect("slot should be available");
        assert!(matches!(ticket.wait(), Err(EvaluationError::Backend(_))));
        evaluator.shutdown();
    }

    #[test]
    fn backend_error_details_reach_waiting_tickets() {
        let evaluator = SharedGpuEvaluator::start(FailingBackend, 1, 1, Duration::ZERO);
        let input = [0.0; INPUT_SIZE];
        let ticket = evaluator
            .submit(&input, PolicyMask::default())
            .expect("slot should be available");
        assert!(matches!(
            ticket.wait(),
            Err(EvaluationError::Backend(message)) if message == "preserved backend detail"
        ));
        evaluator.shutdown();
    }
}

#[cfg(feature = "libtorch")]
mod torchscript {
    use super::{BatchBackend, EvalSlot, EvaluationError};
    use crate::encoding::{INPUT_SIZE, POLICY_SIZE};
    use std::path::Path;
    use tch::{CModule, Device, IValue, Kind, Tensor};

    pub struct TorchScriptBackend {
        module: CModule,
        device: Device,
        input_staging: Vec<f32>,
        mask_staging: Vec<f32>,
        input_pinned: Option<Tensor>,
        mask_pinned: Option<Tensor>,
        policy_host_buffer: Vec<f32>,
        value_host_buffer: Vec<f32>,
        wdl_host_buffer: Vec<f32>,
    }

    impl TorchScriptBackend {
        pub fn load(
            path: impl AsRef<Path>,
            device: Device,
            max_batch: usize,
        ) -> Result<Self, EvaluationError> {
            if max_batch == 0 {
                return Err(EvaluationError::Backend(
                    "TorchScript evaluator max_batch must be positive".to_owned(),
                ));
            }
            let module = CModule::load_on_device(path, device)
                .map_err(|error| EvaluationError::Backend(error.to_string()))?;
            let (input_pinned, mask_pinned) = if matches!(device, Device::Cuda(_)) {
                (
                    Some(
                        Tensor::zeros(
                            [max_batch as i64 * INPUT_SIZE as i64],
                            (Kind::Float, Device::Cpu),
                        )
                        .pin_memory(device),
                    ),
                    Some(
                        Tensor::zeros(
                            [max_batch as i64 * POLICY_SIZE as i64],
                            (Kind::Float, Device::Cpu),
                        )
                        .pin_memory(device),
                    ),
                )
            } else {
                (None, None)
            };
            Ok(Self {
                module,
                device,
                input_staging: vec![0.0; max_batch * INPUT_SIZE],
                mask_staging: vec![0.0; max_batch * POLICY_SIZE],
                input_pinned,
                mask_pinned,
                policy_host_buffer: vec![0.0; max_batch * POLICY_SIZE],
                value_host_buffer: vec![0.0; max_batch],
                wdl_host_buffer: vec![0.0; max_batch * 3],
            })
        }
    }

    impl BatchBackend for TorchScriptBackend {
        fn evaluate_slots(&mut self, slots: &[(usize, &EvalSlot)]) -> Result<(), EvaluationError> {
            let count = slots.len();
            let input_kind = if matches!(self.device, Device::Cuda(_)) {
                Kind::BFloat16
            } else {
                Kind::Float
            };
            for (row, (_, slot)) in slots.iter().enumerate() {
                let data = unsafe { &*slot.data.get() };
                self.input_staging[row * INPUT_SIZE..(row + 1) * INPUT_SIZE]
                    .copy_from_slice(&data.input);
                data.legal.write_f32_mask(
                    &mut self.mask_staging[row * POLICY_SIZE..(row + 1) * POLICY_SIZE],
                );
            }
            let input_source = unsafe {
                Tensor::from_blob(
                    self.input_staging.as_ptr().cast(),
                    &[count as i64 * INPUT_SIZE as i64],
                    &[],
                    Kind::Float,
                    Device::Cpu,
                )
            };
            let mask_source = unsafe {
                Tensor::from_blob(
                    self.mask_staging.as_ptr().cast(),
                    &[count as i64 * POLICY_SIZE as i64],
                    &[],
                    Kind::Float,
                    Device::Cpu,
                )
            };
            let input = if let Some(pinned) = &self.input_pinned {
                let mut host = pinned.narrow(0, 0, count as i64 * INPUT_SIZE as i64);
                host.copy_(&input_source);
                host.to_device(self.device).to_kind(input_kind)
            } else {
                input_source.to_device(self.device).to_kind(input_kind)
            }
            .view([count as i64, crate::encoding::INPUT_CHANNELS as i64, 8, 8]);
            let mask = if let Some(pinned) = &self.mask_pinned {
                let mut host = pinned.narrow(0, 0, count as i64 * POLICY_SIZE as i64);
                host.copy_(&mask_source);
                host.to_device(self.device).to_kind(Kind::Float)
            } else {
                mask_source.to_device(self.device).to_kind(Kind::Float)
            }
            .view([count as i64, POLICY_SIZE as i64]);
            let result = tch::no_grad(|| {
                self.module
                    .forward_is(&[IValue::Tensor(input), IValue::Tensor(mask.shallow_clone())])
            })
            .map_err(|error| EvaluationError::Backend(error.to_string()))?;
            let IValue::Tuple(values) = result else {
                return Err(EvaluationError::Backend(
                    "TorchScript model must return (policy, value, wdl)".into(),
                ));
            };
            if values.len() != 3 {
                return Err(EvaluationError::Backend(
                    "TorchScript model returned the wrong number of tensors".into(),
                ));
            }
            let (IValue::Tensor(policy_logits), IValue::Tensor(value), IValue::Tensor(wdl)) =
                (&values[0], &values[1], &values[2])
            else {
                return Err(EvaluationError::Backend(
                    "TorchScript outputs must be tensors".into(),
                ));
            };
            if policy_logits.size() != [count as i64, POLICY_SIZE as i64]
                || value.size() != [count as i64, 1]
                || wdl.size() != [count as i64, 3]
            {
                return Err(EvaluationError::Backend(format!(
                    "TorchScript output shapes are invalid: policy={:?}, value={:?}, wdl={:?}",
                    policy_logits.size(),
                    value.size(),
                    wdl.size()
                )));
            }
            let policy = policy_logits
                .masked_fill(&mask.le(0), -1.0e4)
                .softmax(-1, Kind::Float)
                .to_device(Device::Cpu)
                .to_kind(Kind::Float);
            let value = value
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .view([count as i64]);
            let wdl = wdl
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .view([count as i64, 3]);

            policy.copy_data(
                &mut self.policy_host_buffer[..count * POLICY_SIZE],
                count * POLICY_SIZE,
            );
            value.copy_data(&mut self.value_host_buffer[..count], count);
            wdl.copy_data(&mut self.wdl_host_buffer[..count * 3], count * 3);

            if self.policy_host_buffer[..count * POLICY_SIZE]
                .iter()
                .chain(self.value_host_buffer[..count].iter())
                .chain(self.wdl_host_buffer[..count * 3].iter())
                .any(|value| !value.is_finite())
            {
                return Err(EvaluationError::Backend(
                    "TorchScript model returned a non-finite output".into(),
                ));
            }

            for (row, (_, slot)) in slots.iter().enumerate() {
                let output = unsafe { &mut (*slot.data.get()).output };
                output.policy.copy_from_slice(
                    &self.policy_host_buffer[row * POLICY_SIZE..(row + 1) * POLICY_SIZE],
                );
                output.value = self.value_host_buffer[row];
                output
                    .wdl
                    .copy_from_slice(&self.wdl_host_buffer[row * 3..(row + 1) * 3]);
            }
            Ok(())
        }
    }
}

#[cfg(feature = "libtorch")]
impl SharedGpuEvaluator {
    pub fn torchscript(
        model_path: impl AsRef<std::path::Path>,
        device: tch::Device,
        slot_count: usize,
        max_batch_size: usize,
        max_wait: Duration,
    ) -> Result<Arc<Self>, EvaluationError> {
        let backend = torchscript::TorchScriptBackend::load(model_path, device, max_batch_size)?;
        Ok(Self::start(backend, slot_count, max_batch_size, max_wait))
    }
}
