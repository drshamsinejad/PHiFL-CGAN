# intialization
from models.mlp import SimpleMLP
from models.cnn import CNN_1
from models.cnn import CNN_2
from models.cnn import CNN_3

def create(dataset,model,loss,metrics,lr,image_shape,number_classes):  
    #if dataset=="mnist":
     #   if model=="mlp":
      #      m=SimpleMLP(784,10,loss,metrics,lr)
            
    if model=='cnn1':
        m=CNN_1(loss,metrics,lr,image_shape,number_classes)  
    elif model=='cnn2':
        m=CNN_2(loss,metrics,lr,image_shape,number_classes)
    elif model=='cnn3':
        m=CNN_3(loss,metrics,lr,image_shape,number_classes)
    return m

            