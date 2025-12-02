from .transformer_student import (
    DistilQwenTransformerStudent,
    DistilVanillaTransformerStudent,
)
from .student import (
    DistilXLSTMStudent,
    StudentArchitectureSpec,
    StudentForwardOutput,
    build_student_spec_from_teacher,
    create_stack_config,
)
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

from .lstm_student import (
    DistilLSTMStudent,
    build_lstm_student_spec_from_teacher,
)

try:
    from .mamba_student import (
        DistilMambaStudent,
        build_mamba_student_spec_from_teacher,
    )
except ImportError:
    print(
        "Mamba student model could not be imported. Ensure all dependencies are installed."
    )


__all__ = [
    "DistilXLSTMStudent",
    "StudentArchitectureSpec",
    "StudentForwardOutput",
    "build_student_spec_from_teacher",
    "create_stack_config",
    "TEACHER_MODEL_ID",
    "TeacherResources",
    "infer_runtime_device",
    "load_teacher_model",
    "run_teacher_smoke_test",
    "XLSTM_MODEL_ID",
    "XLSTMTeacherResources",
    "load_xlstm_teacher",
    "run_xlstm_smoke_test",
    "DistilLSTMStudent",
    "build_lstm_student_spec_from_teacher",
    "DistilQwenTransformerStudent",
    "DistilVanillaTransformerStudent",
]
