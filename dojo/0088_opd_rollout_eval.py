"""Reference and checks for on-policy categorical distillation."""

# %%
import math
import torch
import torch.nn.functional as F

# %%
def opd_rollout(student, teacher, start_tokens, steps, generator, eos_token=None):
    if (not callable(student) or not callable(teacher)
            or not isinstance(start_tokens, torch.Tensor)
            or start_tokens.ndim != 2 or min(start_tokens.shape) < 1
            or start_tokens.dtype != torch.long or torch.any(start_tokens < 0)
            or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or not isinstance(generator, torch.Generator)
            or generator.device.type != start_tokens.device.type):
        raise ValueError('need models, nonempty integer prefixes, steps, and matching generator')
    if eos_token is not None and (not isinstance(eos_token, int) or isinstance(eos_token, bool)):
        raise ValueError('eos_token must be an integer vocabulary index')
    batch = start_tokens.shape[0]
    prefix = start_tokens.clone()
    active = torch.ones(batch, dtype=torch.bool, device=prefix.device)
    total_kl = None
    counts = torch.zeros(batch, dtype=torch.long, device=prefix.device)
    draws = []
    vocab = None
    for time in range(steps):
        if not active.any():
            draws.extend([torch.full((batch,), eos_token, dtype=torch.long, device=prefix.device)
                          for _ in range(steps - time)])
            break
        indices = active.nonzero(as_tuple=True)[0]
        visited = prefix.index_select(0, indices)
        student_logits = student(visited)
        if (not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 2
                or student_logits.shape[0] != len(indices) or student_logits.shape[1] < 2
                or not student_logits.is_floating_point() or student_logits.device != prefix.device
                or not torch.isfinite(student_logits).all()):
            raise ValueError('student must return finite [active,V] logits')
        if vocab is None:
            vocab = student_logits.shape[1]
            if torch.any(start_tokens >= vocab) or (eos_token is not None and not 0 <= eos_token < vocab):
                raise ValueError('prefix or EOS token outside vocabulary')
            total_kl = student_logits.new_zeros(batch)
        elif student_logits.shape[1] != vocab:
            raise ValueError('vocabulary size changed during rollout')
        with torch.no_grad():
            teacher_logits = teacher(visited)
        if (not isinstance(teacher_logits, torch.Tensor)
                or teacher_logits.shape != student_logits.shape
                or teacher_logits.dtype != student_logits.dtype
                or teacher_logits.device != student_logits.device
                or not torch.isfinite(teacher_logits).all()):
            raise ValueError('teacher must return matching finite logits')
        log_student = F.log_softmax(student_logits, dim=-1)
        log_teacher = F.log_softmax(teacher_logits, dim=-1)
        kl = (log_student.exp() * (log_student - log_teacher)).sum(dim=-1)
        total_kl = total_kl.index_add(0, indices, kl)
        counts = counts.index_add(0, indices, torch.ones_like(indices))
        sampled = torch.multinomial(log_student.detach().exp(), 1, generator=generator).squeeze(-1)
        fill = eos_token if eos_token is not None else 0
        token = torch.full((batch,), fill, dtype=torch.long, device=prefix.device)
        token[indices] = sampled
        draws.append(token)
        prefix = torch.cat((prefix, token[:, None]), dim=1)
        if eos_token is not None:
            active = active & token.ne(eos_token)
    tokens = torch.stack(draws, dim=1)
    return tokens, (total_kl / counts).mean()


# %% [verifiers]
class TinyStudent(torch.nn.Module):
    def __init__(self, scores=(0., 0.)):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor(scores, dtype=torch.float64))

    def forward(self, prefix):
        return self.scores.expand(prefix.shape[0], -1)


class RecordingTeacher(torch.nn.Module):
    def __init__(self, scores=(math.log(3.), 0.)):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor(scores, dtype=torch.float64))
        self.seen = []

    def forward(self, prefix):
        self.seen.append(prefix.detach().clone())
        return self.scores.expand(prefix.shape[0], -1)


def verify_baseline_student_prefix_kl_and_grad(opd_rollout):
    student = TinyStudent()
    teacher = RecordingTeacher()
    initial = torch.tensor([[0]], dtype=torch.long)
    tokens, loss = opd_rollout(student, teacher, initial, 2, torch.Generator().manual_seed(3))
    assert tokens.shape == (1, 2) and tokens.dtype == torch.long
    assert torch.allclose(loss, torch.tensor(math.log(2 / math.sqrt(3)), dtype=torch.float64), atol=1e-7)
    assert len(teacher.seen) == 2
    assert torch.equal(teacher.seen[0], initial)
    assert torch.equal(teacher.seen[1], torch.cat([initial, tokens[:, :1]], dim=1))
    loss.backward()
    assert student.scores.grad is not None and torch.count_nonzero(student.scores.grad) == 2
    assert teacher.scores.grad is None


def verify_baseline_seeded_replay(opd_rollout):
    student = TinyStudent((0., 0.))
    start = torch.tensor([[0], [1]], dtype=torch.long)
    a, loss_a = opd_rollout(student, RecordingTeacher(), start, 5, torch.Generator().manual_seed(17))
    b, loss_b = opd_rollout(student, RecordingTeacher(), start, 5, torch.Generator().manual_seed(17))
    assert torch.equal(a, b) and torch.equal(loss_a, loss_b)
    assert torch.equal(start, torch.tensor([[0], [1]]))


def verify_baseline_samples_with_supplied_generator(opd_rollout):
    student = TinyStudent((0., 0.))
    start = torch.zeros((32, 1), dtype=torch.long)
    expected = torch.multinomial(torch.full((32, 2), .5, dtype=torch.float64), 1,
                                 generator=torch.Generator().manual_seed(29))
    tokens, _ = opd_rollout(student, RecordingTeacher(), start, 1,
                            torch.Generator().manual_seed(29))
    assert torch.equal(tokens, expected)
    assert tokens.unique().numel() == 2


def verify_followup_teacher_sees_sampled_not_teacher_prefix(opd_rollout):
    student = TinyStudent((-1000., 0.))
    teacher = RecordingTeacher((1000., 0.))
    tokens, _ = opd_rollout(student, teacher, torch.tensor([[0]]), 2,
                            torch.Generator().manual_seed(1))
    assert torch.equal(tokens, torch.tensor([[1, 1]]))
    assert torch.equal(teacher.seen[1], torch.tensor([[0, 1]]))


def verify_followup_eos_stops_supervision(opd_rollout):
    student = TinyStudent((-1000., 0.))
    teacher = RecordingTeacher((0., 0.))
    start = torch.tensor([[0]])
    tokens, loss = opd_rollout(student, teacher, start, 3,
                               torch.Generator().manual_seed(1), eos_token=1)
    assert torch.equal(tokens, torch.tensor([[1, 1, 1]]))
    assert len(teacher.seen) == 1
    expected = (F.softmax(student.scores, -1)
                * (F.log_softmax(student.scores, -1)
                   - F.log_softmax(teacher.scores.detach(), -1))).sum()
    assert torch.allclose(loss, expected)


def verify_production_invalid_contract(opd_rollout):
    student, teacher = TinyStudent(), RecordingTeacher()
    start = torch.tensor([[0]], dtype=torch.long)
    invalid = [
        (student, teacher, start.float(), 1, torch.Generator(), None),
        (student, teacher, start, 0, torch.Generator(), None),
        (student, teacher, start, 1, torch.Generator(), 2),
        (student, teacher, torch.tensor([[3]]), 1, torch.Generator(), None),
        (TinyStudent((float('nan'), 0.)), teacher, start, 1, torch.Generator(), None),
    ]
    for args in invalid:
        try:
            opd_rollout(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid OPD rollout accepted')


def verify_production_per_completion_length_mean(opd_rollout):
    class PrefixStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([0., 0.], dtype=torch.float64))

        def forward(self, prefix):
            # Row 0 ends immediately; row 1 continues with token 0.
            base = torch.stack((torch.where(prefix[:, 0] == 0, -1000., 1000.),
                                torch.where(prefix[:, 0] == 0, 0., -1000.)), dim=-1)
            return base + self.bias

    class PrefixTeacher(RecordingTeacher):
        def forward(self, prefix):
            self.seen.append(prefix.detach().clone())
            scores = torch.where(prefix[:, :1] == 0, 0., math.log(3.)).to(torch.float64)
            return torch.cat((scores, torch.zeros_like(scores)), dim=1)

    student = PrefixStudent()
    teacher = PrefixTeacher()
    tokens, loss = opd_rollout(student, teacher, torch.tensor([[0], [1]]), 3,
                               torch.Generator().manual_seed(5), eos_token=1)
    assert torch.equal(tokens, torch.tensor([[1, 1, 1], [0, 0, 0]]))
    assert [prefix.shape[0] for prefix in teacher.seen] == [2, 1, 1]
    expected = (math.log(2.) + math.log(4. / 3.)) / 2.
    assert torch.allclose(loss, torch.tensor(expected, dtype=torch.float64), atol=1e-7)
    loss.backward()
    assert student.bias.grad is not None and torch.isfinite(student.bias.grad).all()


# %%
from torch_judge import evaluate
if not evaluate('0088_opd_rollout_eval.py', opd_rollout):
    raise SystemExit(1)
