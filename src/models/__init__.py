from .teacher import (
    TEACHER_MODEL_ID,
    TeacherResources,
    infer_runtime_device,
    load_teacher_model,
    run_teacher_smoke_test,
)
from .xlstm_teacher import (
    XLSTM_MODEL_ID,
    XLSTMTeacherResources,
    load_xlstm_teacher,
    run_xlstm_smoke_test,
)

__all__ = [
    "TEACHER_MODEL_ID",
    "TeacherResources",
    "infer_runtime_device",
    "load_teacher_model",
    "run_teacher_smoke_test",
    "XLSTM_MODEL_ID",
    "XLSTMTeacherResources",
    "load_xlstm_teacher",
    "run_xlstm_smoke_test",
]
