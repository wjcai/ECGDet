from prepropossing import *
from loss import *
from model import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
if __name__ == '__main__':
    datapath='data//118'#Take the 118th data of MITDB as an example
    model=model_build()
    model_checkpoint = ModelCheckpoint(
            'weights_kfold_v.h5', monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(
            monitor='val_loss', patience=3, verbose=1)
    reducel = ReduceLROnPlateau(
            monitor='val_loss', patience=2, verbose=1, factor=0.1,mode='min')
    
    history = model.fit_generator(data_gen(train_index,batch=128,aug=True,gen=y_gen2),
                              steps_per_epoch=46,
                              epochs=50,
                              validation_data=data_gen(val_index,batch=128,aug=False,gen=y_gen2),
                              validation_steps=12,
                                      callbacks=[reducel,earlystopper,model_checkpoint, tensorboard])




