from typing import Any

from .base_models import (
    BaseTorchModel,
    BaseRawModel,
    ForwardMode,
    BaseOptimizationModel,
    AbstractBatcher,
)


def construct_model(name: str, args: dict[str, Any]) -> BaseTorchModel:
    name = name.lower()
    if name not in name_to_class_lookup:
        raise ValueError(f"Unknown model name: {name}")

    cls = name_to_class_lookup[name]
    return cls(**args)


from .constant_vector_baseline import ConstantVectorBaseline
from .feed_forward.deflow import DeFlow
from .feed_forward.flow4d import Flow4D
from .feed_forward.fast_flow_3d import (
    FastFlow3D,
    FastFlow3DBucketedLoaderLoss,
    FastFlow3DSelfSupervisedLoss,
)
from .feed_forward.flow4d import Flow4D
from .whole_batch_optimization import (
    NSFPForwardOnlyOptimizationLoop,
    NSFPCycleConsistencyOptimizationLoop,
    FastNSFModelOptimizationLoop,
    Liu2024OptimizationLoop,
)
from .mini_batch_optimization import (
    EulerFlowOptimizationLoop,
    NTPOptimizationLoop,
    EulerFlowSincOptimizationLoop,
    EulerFlowGaussianOptimizationLoop,
    EulerFlowFourtierOptimizationLoop,
    EulerFlowDepth22OptimizationLoop,
    EulerFlowDepth20OptimizationLoop,
    EulerFlowDepth18OptimizationLoop,
    EulerFlowDepth16OptimizationLoop,
    EulerFlowDepth14OptimizationLoop,
    EulerFlowDepth12OptimizationLoop,
    EulerFlowDepth10OptimizationLoop,
    EulerFlowDepth8OptimizationLoop,
    EulerFlowDepth6OptimizationLoop,
    EulerFlowDepth4OptimizationLoop,
    EulerFlowDepth2OptimizationLoop,
    EulerFlowNoCycleConsistencyLossOptimizationLoop,
    EulerFlowNoKStepLossOptimizationLoop,
)


importable_models = [
    DeFlow,
    Flow4D,
    FastFlow3D,
    Flow4D,
    ConstantVectorBaseline,
    NSFPForwardOnlyOptimizationLoop,
    NSFPCycleConsistencyOptimizationLoop,
    FastNSFModelOptimizationLoop,
    Liu2024OptimizationLoop,
    EulerFlowOptimizationLoop,
    EulerFlowSincOptimizationLoop,
    NTPOptimizationLoop,
    EulerFlowGaussianOptimizationLoop,
    EulerFlowFourtierOptimizationLoop,
    EulerFlowDepth22OptimizationLoop,
    EulerFlowDepth20OptimizationLoop,
    EulerFlowDepth18OptimizationLoop,
    EulerFlowDepth16OptimizationLoop,
    EulerFlowDepth14OptimizationLoop,
    EulerFlowDepth12OptimizationLoop,
    EulerFlowDepth10OptimizationLoop,
    EulerFlowDepth8OptimizationLoop,
    EulerFlowDepth6OptimizationLoop,
    EulerFlowDepth4OptimizationLoop,
    EulerFlowDepth2OptimizationLoop,
    EulerFlowNoCycleConsistencyLossOptimizationLoop,
    EulerFlowNoKStepLossOptimizationLoop,
]

# Ensure all importable models are based on the BaseModel class.
for cls in importable_models:
    assert issubclass(cls, BaseTorchModel) or issubclass(
        cls, BaseRawModel
    ), f"{cls} is not a valid model class."

name_to_class_lookup = {cls.__name__.lower(): cls for cls in importable_models}
