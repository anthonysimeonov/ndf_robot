from yacs.config import CfgNode as CN
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet


class MultiCams:
    """
    Class for easily obtaining simulated camera image observations in pybullet
    """
    def __init__(self, cam_cfg, pb_client, n_cams=2):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            robot (airobot Robot): Instance of PyBullet simulated robot, from
                airobot library
            n_cams (int): Number of cameras to put in the world
        """
        super(MultiCams, self).__init__()
        self.n_cams = n_cams
        self.cams = []
        self.cfg = cam_cfg
        self.pb_client = pb_client
        self.focus_pt = self.cfg.FOCUS_PT
        for _ in range(n_cams):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs(),
                                                pb_client=pb_client))

        self.cam_setup_cfg = {}
        self.cam_setup_cfg['focus_pt'] = [self.cfg.FOCUS_PT] * self.n_cams
        self.cam_setup_cfg['dist'] = [self.cfg.DISTANCE] * self.n_cams
        self.cam_setup_cfg['yaw'] = self.cfg.YAW_ANGLES[:self.n_cams]
        self.cam_setup_cfg['pitch'] = [self.cfg.PITCH] * self.n_cams
        self.cam_setup_cfg['roll'] = [0] * self.n_cams

        self._setup_cameras()

    def _camera_cfgs(self):
        """
        Returns a set of camera config parameters

        Returns:
            YACS CfgNode: Cam config params
        """
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 10
        _C.WIDTH = 640
        _C.HEIGHT = 480
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        return _ROOT_C.clone()

    def _setup_cameras(self):
        """
        Function to set up multiple pybullet cameras in the simulated environment
        """
        for i, cam in enumerate(self.cams):
            cam.setup_camera(
                focus_pt=self.cam_setup_cfg['focus_pt'][i],
                dist=self.cam_setup_cfg['dist'][i],
                yaw=self.cam_setup_cfg['yaw'][i],
                pitch=self.cam_setup_cfg['pitch'][i],
                roll=self.cam_setup_cfg['roll'][i]
            )
