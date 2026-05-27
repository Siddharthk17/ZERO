"""Exponential moving average teacher network."""

from __future__ import annotations

import copy

import torch


class EMATeacher:
    def __init__(self, student: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        teacher_state = self.teacher.state_dict()
        student_state = student.state_dict()
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)

    @torch.no_grad()
    def promote(self, student: torch.nn.Module) -> None:
        self.teacher.load_state_dict(student.state_dict())
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)


@torch.no_grad()
def update_ema_teacher(student: torch.nn.Module, teacher: torch.nn.Module, decay: float = 0.999) -> None:
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)
