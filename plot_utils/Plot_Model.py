def plotModel(history): 
  fig, axs = plt.subplots(1,2,figsize=(16,5)) 
  # summarize history for accuracy
  axs[0].plot(history.history['sparse_categorical_accuracy'], 'c') 
  axs[0].plot(history.history['val_sparse_categorical_accuracy'],'m') 
  axs[0].set_title('Model Accuracy')
  axs[0].set_ylabel('Accuracy') 
  axs[0].set_xlabel('Epoch')
  axs[0].legend(['train', 'validate'], loc='upper left')
  # summarize history for loss
  axs[1].plot(history.history['loss'], 'c') 
  axs[1].plot(history.history['val_loss'], 'm') 
  axs[1].set_title('Model Loss')
  axs[1].set_ylabel('Loss') 
  axs[1].set_xlabel('Epoch')
  axs[1].legend(['train', 'validate'], loc='upper right')
  plt.show()
