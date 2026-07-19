"""Local-first Chinese PII cascade library."""

from .ablation_profiles import (
    REQUIRED_ABLATION_IDS,
    AblationRuntimeProfile,
    load_ablation_profile,
    load_ablation_profiles,
)
from .community_model import (
    COMMUNITY_MODEL_IDENTITY_SCHEMA_VERSION,
    CommunityModelContractError,
    CommunityModelIdentity,
    VerifiedCommunityModel,
    expected_core24_label2id,
    verify_community_model_artifact,
)
from .config import CASCADE_CONFIG_SCHEMA_VERSION, CascadeConfig, CascadeMode
from .context import ContextCandidate, ContextDecision, ContextEnhancer, ContextOutcome
from .pipeline import CascadePipeline, Recognizer, Validator
from .result import CASCADE_DETECTION_SCHEMA_VERSION, CascadeDetection
from .routing import (
    ENTITY_ALIASES,
    EntityRoute,
    RoutingDecision,
    canonicalize_entity_type,
    community_full24_routes,
    conservative_v2_routes,
    default_routes,
)
from .service_profiles import (
    COMMUNITY_MODEL_SERVICE_PROFILE_VERSION,
    DEFAULT_SERVICE_PROFILE_VERSION,
    LEGACY_SERVICE_PROFILE_VERSION,
    SERVICE_PROFILE_MODE_MATRIX,
    SERVICE_PROFILE_VERSIONS,
    SUCCESSOR_SERVICE_PROFILE_VERSION,
    build_community_model_service_pipeline,
    build_rules_only_service_pipeline,
    load_service_config,
    validate_service_profile_mode,
)
from .stages import (
    DEFAULT_RELEASE_STAGE_POLICY,
    PIPELINE_STAGE_POLICY_SCHEMA_VERSION,
    PipelineStagePolicy,
)

__all__ = [
    "CASCADE_CONFIG_SCHEMA_VERSION",
    "CASCADE_DETECTION_SCHEMA_VERSION",
    "COMMUNITY_MODEL_SERVICE_PROFILE_VERSION",
    "COMMUNITY_MODEL_IDENTITY_SCHEMA_VERSION",
    "DEFAULT_RELEASE_STAGE_POLICY",
    "DEFAULT_SERVICE_PROFILE_VERSION",
    "ENTITY_ALIASES",
    "LEGACY_SERVICE_PROFILE_VERSION",
    "PIPELINE_STAGE_POLICY_SCHEMA_VERSION",
    "REQUIRED_ABLATION_IDS",
    "SERVICE_PROFILE_VERSIONS",
    "SERVICE_PROFILE_MODE_MATRIX",
    "SUCCESSOR_SERVICE_PROFILE_VERSION",
    "AblationRuntimeProfile",
    "CascadeConfig",
    "CascadeDetection",
    "CascadeMode",
    "CascadePipeline",
    "ContextCandidate",
    "ContextDecision",
    "ContextEnhancer",
    "ContextOutcome",
    "CommunityModelContractError",
    "CommunityModelIdentity",
    "EntityRoute",
    "PipelineStagePolicy",
    "Recognizer",
    "RoutingDecision",
    "Validator",
    "VerifiedCommunityModel",
    "build_community_model_service_pipeline",
    "build_rules_only_service_pipeline",
    "canonicalize_entity_type",
    "community_full24_routes",
    "conservative_v2_routes",
    "default_routes",
    "load_ablation_profile",
    "load_ablation_profiles",
    "load_service_config",
    "expected_core24_label2id",
    "validate_service_profile_mode",
    "verify_community_model_artifact",
]
