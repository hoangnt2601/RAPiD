import numpy as np
import onnxruntime as rt
import cv2
import onnx


def preprocess(img, shape=(64, 128)):
	img = cv2.resize(img, shape)
	img = np.float32(img)
	img = img / 255.0
	img = img.transpose(2, 1, 0)
	img = np.expand_dims(img, axis=0)

	return img


class Extractor:
	def __init__(self, model_path) -> None:
		model = onnx.load(model_path)
		self.engine = rt.InferenceSession(model_path)
		self.input_names = [node.name for node in model.graph.input]
		self.output_names = [node.name for node in model.graph.output]

	def __call__(self, im_crops):
		embs = []
		for im in im_crops:
			inp = preprocess(im, (256, 128))
			emb = self.engine.run(self.output_names, {self.input_names[0]: inp})[0]
			embs.append(emb.squeeze())
		embs = np.array(np.stack(embs), dtype=np.float32)
		return embs


if __name__ == "__main__":
	img = cv2.imread("img.jpg")
	extr = Extractor("/workspace/deep_sort_pytorch/pretrained/osnet_x0_25_msmt17_dynamic.onnx")
	feature = extr([img])
	print(feature.shape)
