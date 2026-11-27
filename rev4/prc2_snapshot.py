"""Fix the GT lifetime bug without changing PRC2 thresholds, sampling or loss.

Installed DBModuleLoss.get_targets shifts polygon arrays in-place in _draw_border_map.
The original PRC2 read those modified arrays AFTER detection loss; its preflight read
them beforehand. Snapshot the aligned sample before calling the detection loss.
The original PRC2 source and failed run remain unchanged for audit.
"""
import copy
from mmocr.registry import MODELS
from rev4 import require
from rev4.prc2_plugins import Rev4PRC2DBHead
from rev4.prc_geometry import masked_response_loss


@MODELS.register_module()
class Rev4PRC2SnapshotDBHead(Rev4PRC2DBHead):
    def loss(self,x,batch_data_samples):
        require([s.get('condition') for s in batch_data_samples]==['clean','lr2'],
                'PRC2 snapshot requires the unchanged B1 microbatch')
        aligned=copy.deepcopy(batch_data_samples[1])
        outputs=self(x,batch_data_samples,mode='both')
        losses=self.module_loss(outputs[:3],batch_data_samples)
        target,mask,selected,eligible=self.teacher_target(aligned)
        student=outputs[3][1]
        require(student.shape==target.shape,'PRC2 snapshot probability grids differ')
        losses['loss_prc2']=self.prc2['weight']*masked_response_loss(student,target,mask)
        losses['prc2_pixels']=mask.sum().float()
        losses['prc2_selected']=student.new_tensor(float(selected))
        losses['prc2_eligible']=student.new_tensor(float(eligible))
        return losses
