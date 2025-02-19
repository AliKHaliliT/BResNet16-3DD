import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from BResNet163DD import BResNet163DD


model = BResNet163DD()
model.build((None, 32, 256, 256, 3))
model.summary()