from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.skill_cooldown.SkillCooldown import SkillCooldown
from policy.skill_cast.SkillCast import SkillCast
from policy.beyond_mimic.BeyondMimic import BeyondMimic
from FSM.FSMState import FSMState, FSMStateName
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
import time
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput, 
                 enable_logging: bool = False, log_dir: str = "./logs", log_prefix: str = "", robot_uid: int = None):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        
        # Instantiate all policies
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.skill_cast_policy = SkillCast(state_cmd, policy_output)
        # BeyondMimic with optional logging support
        self.beyond_mimic_policy = BeyondMimic(state_cmd, policy_output, 
                                               enable_logging=enable_logging, 
                                               log_dir=log_dir,
                                               prefix=log_prefix,
                                               robot_uid=robot_uid)
        # BeyondMimic with optional logging support
        self.beyond_mimic_policy = BeyondMimic(state_cmd, policy_output, 
                                               enable_logging=enable_logging, 
                                               log_dir=log_dir,
                                               prefix=log_prefix,
                                               robot_uid=robot_uid)
        
        # Map from state name to policy instance
        self._states = {
            FSMStateName.PASSIVE: self.passive_mode,
            FSMStateName.FIXEDPOSE: self.fixed_pose_1,
            FSMStateName.LOCOMODE: self.loco_policy,
            FSMStateName.SKILL_COOLDOWN: self.skill_cooldown_policy,
            FSMStateName.SKILL_CAST: self.skill_cast_policy,
            FSMStateName.SKILL_BEYOND_MIMIC: self.beyond_mimic_policy,
        }

        # Table-driven external command transitions
        self._cmd_transitions = {
            FSMStateName.PASSIVE: {
                FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE,
            },
            FSMStateName.FIXEDPOSE: {
                FSMCommand.LOCO: FSMStateName.LOCOMODE,
                FSMCommand.SKILL_4: FSMStateName.SKILL_BEYOND_MIMIC,
                FSMCommand.PASSIVE: FSMStateName.PASSIVE,
            },
            FSMStateName.LOCOMODE: {
                FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE,
                FSMCommand.SKILL_4: FSMStateName.SKILL_BEYOND_MIMIC,
                FSMCommand.PASSIVE: FSMStateName.PASSIVE,
            },
            FSMStateName.SKILL_BEYOND_MIMIC: {
                FSMCommand.LOCO: FSMStateName.SKILL_COOLDOWN,
                FSMCommand.PASSIVE: FSMStateName.PASSIVE,
                FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE,
            },
            FSMStateName.SKILL_COOLDOWN: {
                FSMCommand.PASSIVE: FSMStateName.PASSIVE,
            },
        }

        print("initalized all policies!!!")
        
        self.cur_state_name = FSMStateName.PASSIVE
        self.cur_policy = self._states[self.cur_state_name]
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        start_time = time.time()

        # Run current policy control
        self.cur_policy.run()

        # External command transition (table-driven)
        cmd = self.state_cmd.skill_cmd
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        next_name = self.cur_state_name
        if cmd != FSMCommand.INVALID:
            trans = self._cmd_transitions.get(self.cur_state_name, {})
            next_name = trans.get(cmd, self.cur_state_name)

        # Internal transition (state-specific, e.g., motion finished)
        internal_next = self.cur_policy.internal_check()
        if internal_next is not None:
            next_name = internal_next

        # Switch state if needed
        if next_name != self.cur_state_name:
            self.cur_policy.exit()
            self.cur_state_name = next_name
            self.cur_policy = self._states[self.cur_state_name]
            self.cur_policy.enter()
            print("Switched to ", self.cur_policy.name_str)

        end_time = time.time()
        # print("time cusume: ", end_time - start_time)

    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        """
        Kept for backward compatibility; prefer using cur_state_name +
        internal table instead of calling this directly.
        """
        if policy_name in self._states:
            self.cur_state_name = policy_name
            self.cur_policy = self._states[policy_name]
