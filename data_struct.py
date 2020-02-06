
class CompData:
    def __init__(self):
        self.mft_score = None
        self.affect_side = None

        self.reach_begin = None
        self.reach_end = None
        self.retract_begin = None
        self.retract_end = None

        self.reach_comp_score = None
        self.retract_comp_score = None
        self.reach_comp_label = None
        self.retract_comp_label = None
        self.reach_fas_score = None
        self.retract_fas_score = None

        self.target_dist = None
        self.target_angle = None

        self.free_acc_L_fore_reach = []
        self.free_acc_L_fore_retract = []

        self.free_acc_R_fore_reach = []
        self.free_acc_R_fore_retract = []

        self.free_acc_L_upper_reach = []
        self.free_acc_L_upper_retract = []

        self.free_acc_R_upper_reach = []
        self.free_acc_R_upper_retract = []

        self.free_acc_trunk_reach = []
        self.free_acc_trunk_retract = []

        self.free_acc_affect_fore_reach = []
        self.free_acc_affect_fore_retract = []
        self.free_acc_affect_upper_reach = []
        self.free_acc_affect_upper_retract = []

        self.vel_affect_fore_reach = []
        self.vel_affect_fore_retract = []

        self.velfilt_affect_fore_reach = []
        self.velfilt_affect_fore_retract = []

        self.accfilt_affect_fore_reach = []
        self.accfilt_affect_fore_retract = []



