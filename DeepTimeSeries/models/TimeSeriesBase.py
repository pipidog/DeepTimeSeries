import pickle 
from keras.optimizers import Adam

class TSBase:
    def summary(self):
        return self.model.summary()
    
    def compile(self, optimizer=Adam(0.001), loss='mse', metrics='mse'):
        self.model.compile(optimizer=optimizer, loss='mse')
    
    def save(self,model_name):
        self.model.save(model_name+'.h5')
        pickle.dump(self.class_info,open(model_name+'.info','wb'))

    def data_preprocessing(self,x, y = None):
        # overload this method if it doesn't fit your need. 
        x_new = x
        y_new = y

        return x_new, y_new

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, validation_data=None):
        x_train, y_train = self.data_preprocessing(x, y)
        if validation_data != None:
            x_test, y_test = self.data_preprocessing(validation_data[0], validation_data[1])
            validation_data = (x_test, y_test)

        self.history=self.model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=False)

        return self.history
    
    def predict(self, x_test):
        x_test, _ = self.data_preprocessing(x_test)
        y_predict = self.model.predict(x_test)
        
        return y_predict