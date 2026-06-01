"""Exponential moving average teacher network utilities with type and device safety."""

from __future__ import annotations

import copy
import torch

class EMATeacher:
    """Manages an Exponential Moving Average (EMA) teacher network to stabilize reinforcement learning."""

    def __init__(self, student: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        """Saves weights from the student network into the teacher using EMA math."""
        teacher_state = self.teacher.state_dict()
        student_state = student.state_dict()
        
        for name, teacher_value in teacher_state.items():
            if name not in student_state:
                continue
            
            # Align student parameters to the teacher's device and dtype to prevent crashes
            student_value = student_state[name].detach().to(
                device=teacher_value.device, 
                dtype=teacher_value.dtype
            )
            
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)

    @torch.no_grad()
    def promote(self, student: torch.nn.Module) -> None:
        """Instantly align all teacher weights to match the student network."""
        self.teacher.load_state_dict(student.state_dict())
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

@torch.no_grad()
def update_ema_teacher(student: torch.nn.Module, teacher: torch.nn.Module, decay: float = 0.999) -> None:
    """Standalone helper function to update any EMA target model."""
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    
    for name, teacher_value in teacher_state.items():
        if name not in student_state:
            continue
            
        student_value = student_state[name].detach().to(
            device=teacher_value.device, 
            dtype=teacher_value.dtype
        )
        
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)