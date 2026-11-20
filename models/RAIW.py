import torch.nn as nn


class RAIWModel(nn.Module):
    """Wrapper for `wiltodelta/raiw-source-classify` (`remove-ai-watermarks[source-classify]`)."""

    def __init__(self):
        super(RAIWModel, self).__init__()
        self.weights_path = None

    def load_weights(self, ckpt):
        if ckpt and ckpt not in ("hub", "default", "./weights/univfd/fc_weights.pth"):
            self.weights_path = ckpt

    def predict(self, imgs):
        from remove_ai_watermarks import source_classify as sc

        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        preds = []
        for img in imgs:
            res = sc.classify_source(img, weights_path=self.weights_path)
            prob = float(res.synthetic_probability)
            if res.synthetic_detected and prob <= 0.5:
                prob = 0.500001
            elif not res.synthetic_detected and prob > 0.5:
                prob = 0.499999
            preds.append(prob)
        return preds
