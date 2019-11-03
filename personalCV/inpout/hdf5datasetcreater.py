import h5py
import os

class DatsetWriter:

	def __init__(self, dimmension, path_of_output, dataKey="images",bufferSize=1500):
		if os.path.exists(path_of_output):
			raise ValueError("The output directory(or files) already exists.", path_of_output)
		self.db = h5py.File(path_of_output, "w")
		self.data = self.db.create_dataset(dataKey, dimmension,dtype="float")
		self.labels = self.db.create_dataset("labels", (dimmension[0],),dtype="int")
		self.bufferSize = bufferSize
		self.buffer = {"data": [], "labels": []}
		self.idx = 0


	def add(self, rows, labels):
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)		
		if len(self.buffer["data"]) >= self.bufferSize:
			self.flush()

	def flush(self):
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}		


	def store_labels_of_classes(self,label_of_classes):
		dt = h5py.special_dtype(vlen=unicode)
		label_saver = self.db.create_dataset("label_names",(len(label_of_classes),), dtype=dt)
		label_saver[:] = label_of_classes

	def close(self):
		if len(self.buffer["data"]) > 0:
			self.flush()
		self.db.close()	

		
