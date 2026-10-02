import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from gmr_npz import export_qpos, validate, JOINT_NAMES
from verify_npz import check, wrist_quality
from general_motion_retargeting import ROBOT_XML_DICT
import mujoco


class ExportTest(unittest.TestCase):
    def test_rejects_extreme_or_sustained_saturated_wrist(self):
        dof = np.zeros((60, 29))
        wrist_quality(dof)
        left_pitch = JOINT_NAMES.index('left_wrist_pitch_joint')
        dof[:, left_pitch] = np.deg2rad(92.5)
        with self.assertRaisesRegex(AssertionError, 'range exceeded'):
            wrist_quality(dof)
        dof[:, left_pitch] = np.deg2rad(29.5)
        with self.assertRaisesRegex(AssertionError, 'sustained'):
            wrist_quality(dof)

    def test_rotated_root_fk_and_signs(self):
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT['unitree_g1']))
        qpos = np.tile(model.qpos0, (8, 1))
        qpos[:, :3] = [2, -3, 1]
        qpos[:, 3:7] = [np.cos(.4), 0, 0, np.sin(.4)]
        qpos[1::2, 3:7] *= -1
        qpos[:, 7:] = (model.jnt_range[1:, 0] + model.jnt_range[1:, 1]) / 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.npz'
            a = export_qpos(qpos, 30., model, path)
            self.assertEqual(tuple(a['joint_names']), JOINT_NAMES)
            np.testing.assert_array_equal(a['root_pos'][0, :2], [0, 0])
            self.assertLess(check(path)['fk_max_error_m'], 1e-5)
            with np.load(path, allow_pickle=False) as z:
                validate(dict(z))
            for name, invalid in [('fps', np.asarray(float('nan'))),
                                  ('dof_pos', np.zeros((8,28))),
                                  ('link_body_list', np.array([None], dtype=object))]:
                with self.assertRaises(ValueError):
                    validate({**a, name: invalid})

if __name__ == '__main__':
    unittest.main()
