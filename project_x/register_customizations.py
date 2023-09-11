from argparse import Action

from project_x.obs.DefaultWithTimeoutsObsBuilder import DefaultWithTimeoutsObsBuilder
from project_x.obs.PaddedDefaultWithTimeoutsObsBuilder import PaddedDefaultWithTimeoutsObsBuilder
from project_x.obs.AbsoluteUnitObs import AbsoluteUnitObs

from project_x.rewards.GoalVelocityReward import GoalVelocityReward
from project_x.rewards.LogCombinedReward import LogCombinedReward
from project_x.rewards.LogEventReward import LogEventReward
from project_x.rewards.LogJumpTouchReward import LogJumpTouchReward
from project_x.rewards.LogTouchHeightReward import LogTouchHeightReward
from project_x.rewards.AerialDistanceReward import AerialDistanceReward

from project_x.terminal_conditions.aerial_task_terminal_condition import AerialTaskTerminalCondition

from rlgym_distrib_rl_wrapper.ObsBuilderFactory import build_obs_builder_from_config

from rlgym_distrib_rl_wrapper import ActionParserFactory, \
    ObsBuilderFactory, RewardFunctionFactory, StateSetterFactory, \
    TerminalConditionsFactory

from rlgym_tools.extra_obs.general_stacking import GeneralStacker

def register_custom_action_parsers():
    # example action parser registration:
    # ActionParserFactory.register_action_parser("custom_action_parser", CustomActionParser)
    pass

def register_custom_obs_builders():
    general_stacker_arg_transformer =  lambda **kwargs: {
        "obs": build_obs_builder_from_config(kwargs["obs"]),
        "stack_size": kwargs.get("stack_size", 15)
    }
    ObsBuilderFactory.register_obs_builder("default_with_timeouts", DefaultWithTimeoutsObsBuilder)
    ObsBuilderFactory.register_obs_builder("padded_default_with_timeouts", PaddedDefaultWithTimeoutsObsBuilder)
    ObsBuilderFactory.register_obs_builder("absolute_unit_obs", AbsoluteUnitObs)
    ObsBuilderFactory.register_obs_builder("tools_general_stacker", GeneralStacker, args_transformer=general_stacker_arg_transformer)

def register_custom_reward_functions():
    log_combine_args_transformer = RewardFunctionFactory._arg_transformers["combined"]
    RewardFunctionFactory.register_reward_function("log_combined", LogCombinedReward, args_transformer=log_combine_args_transformer)
    RewardFunctionFactory.register_reward_function("log_event", LogEventReward)
    RewardFunctionFactory.register_reward_function("log_jump_touch", LogJumpTouchReward)
    RewardFunctionFactory.register_reward_function("log_touch_height", LogTouchHeightReward)
    RewardFunctionFactory.register_reward_function("goal_velocity", GoalVelocityReward)
    RewardFunctionFactory.register_reward_function("rolv_aerial", AerialDistanceReward)

def register_custom_state_setters():
    # example state setter registration:
    # StateSetterFactory.register_state_setter("custom_state_setter", CustomStateSetter)
    pass

def register_custom_terminal_conditions():
    TerminalConditionsFactory.register_terminal_condition("aerial_task", AerialTaskTerminalCondition)

def register_customizations():
    register_custom_action_parsers()
    register_custom_obs_builders()
    register_custom_state_setters()
    register_custom_reward_functions()
    register_custom_terminal_conditions()
