import warnings

# 忽略 albumentations 检查版本时网络超时的无害警告
warnings.filterwarnings(
    "ignore",
    message="Error fetching version info",
    category=UserWarning,
    module=r"albumentations.*",
)
