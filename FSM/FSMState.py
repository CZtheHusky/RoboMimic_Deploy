from common.path_config import PROJECT_ROOT

from common.utils import FSMStateName    

class FSMState:
    def __init__(self):
        self.name = FSMStateName.INVALID
        self.name_str = "invalid"
        self.control_dt = 0.02

    def enter(self):
        raise NotImplementedError("enter() function must be implement!")
    
    def run(self):
        raise NotImplementedError("run() function must be implement!")
    
    def exit(self):
        raise NotImplementedError("exit() function must be implement!")
    
    def internal_check(self):
        """
        State-internal transition logic that does not depend on external
        commands (for example, motion finished).
        Should return a FSMStateName or None.
        """
        return None
        
