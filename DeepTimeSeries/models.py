import numpy as np
import pickle 
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Dropout, Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K

class TimeSeriesBase:
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

class RNN2Dense(TimeSeriesBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, 
        dense_units = None, dropout_rate = 0.3, reload = False ):
        """
            input_shape: shape of input data, (n_memory_steps, n_in_features)
            output_shape: shape of output data, (n_forcast_steps, n_out_features)
            cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
            cell_units: number of hidden cell unit in RNN part, integer, e.g. 100
            dense_units: units of the hidden dense layers, a tuple, e.g, (20,30)
            reload: True if feed externally, False if generate from scratch 
        """

        if reload:
            self.model = None
            self.class_info = None
        else:
            # just for future save
            self.class_info={'class': 'RNN2Dense', 'input_shape': input_shape, 
                'output_shape': output_shape, 'cell': cell, 'cell_units': cell_units,
                'dense_units': dense_units} 

            # check input variables 
            assert cell in ['SimpleRNN', 'LSTM', 'GRU']
            assert type(cell_units) == int
            assert type(dense_units) == tuple
            
            # batch_input_shape = [n_sample, n_backtrack_step, n_feature] 
            x_in = Input(input_shape)
            if cell == 'SimpleRNN':
                x = SimpleRNN(units=cell_units)(x_in)
            elif cell == 'LSTM':
                x = LSTM(units=cell_units)(x_in)
            elif cell == 'GRU':
                x = GRU(units=cell_units)(x_in)
            if dense_units != None:
                for n_units in dense_units:
                    x = Dense(n_units, activation='relu')(x)
                    x = Dropout(dropout_rate)(x)
            x = Dense(np.prod(output_shape))(x)
            x_out = Reshape((output_shape))(x)
            self.model = Model(inputs = x_in, outputs = x_out)



class Seq2Seq_1(TimeSeriesBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, reload = False ):
        """
            input_shape: shape of input data, (n_memory_steps, n_in_features)
            output_shape: shape of output data, (n_forcast_steps, n_out_features)
            cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
            cell_units: number of hidden cell unit in RNN part, integer, e.g. 100
            reload: True if feed externally, False if generate from scratch 
        """
        if reload:
            self.model = None
            self.class_info = None
        else:
            # just for future reload
            self.class_info = {'class': 'Seq2Seq', 'input_shape': input_shape, 'output_shape': output_shape,
                'cell': cell, 'cell_units': cell_units}

            if cell == 'LSTM':
                # declare encoder and decoder objects
                encoder = LSTM(units = cell_units, return_state = True)
                decoder = LSTM(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h, state_c = encoder(encoder_input)
                encoder_state = [state_h, state_c]  

                decoder_input = Input((1,output_shape[-1]))
                
                # initial input and state for iteration
                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = [state_h, state_c]

            elif cell == 'SimpleRNN':
                # declare encoder and decoder objects
                encoder = SimpleRNN(units = cell_units, return_state = True)
                decoder = SimpleRNN(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  

                decoder_input = Input((1,output_shape[-1]))
                
                # initial input and state for iteration
                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = state_h

            elif cell == 'GRU':
                # declare encoder and decoder objects
                encoder = GRU(units = cell_units, return_state = True)
                decoder = GRU(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  

                decoder_input = Input((1,output_shape[-1]))
                
                # initial input and state for iteration
                iter_input = decoder_input
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = state_h

            # Concatenate all predictions
            decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
            self.model = Model([encoder_input, decoder_input], decoder_output)

    def data_preprocessing(self, x, y = None):
        x_new = [x, np.ones((len(x), 1, self.class_info['output_shape'][1]))]
        y_new = y

        return x_new, y_new


class Seq2Seq_2(TimeSeriesBase):
    def __init__(self, input_shape, output_shape, cell, cell_units, reload = False ):
        """
            input_shape: shape of input data, (n_memory_steps, n_in_features)
            output_shape: shape of output data, (n_forcast_steps, n_out_features)
            cell: cell in the RNN part, 'SimpleRNN' / 'LSTM' / 'GRU'
            cell_units: number of hidden cell unit in RNN part, integer, e.g. 100
            reload: True if feed externally, False if generate from scratch 
        """
        if reload:
            self.model = None
            self.class_info = None
        else:
            # just for future reload
            self.class_info = {'class': 'Seq2Seq', 'input_shape': input_shape, 'output_shape': output_shape,
                'cell': cell, 'cell_units': cell_units}

            if cell == 'LSTM':
                # declare encoder and decoder objects
                encoder = LSTM(units = cell_units, return_state = True)
                decoder = LSTM(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h, state_c = encoder(encoder_input)
                encoder_state = [state_h, state_c]  
                
                # initial input and state for iteration
                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h, state_c = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = [state_h, state_c]

            elif cell == 'SimpleRNN':
                # declare encoder and decoder objects
                encoder = SimpleRNN(units = cell_units, return_state = True)
                decoder = SimpleRNN(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  
                
                # initial input and state for iteration
                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = state_h

            elif cell == 'GRU':
                # declare encoder and decoder objects
                encoder = GRU(units = cell_units, return_state = True)
                decoder = GRU(units = cell_units, return_sequences=True, return_state = True)
                decoder_dense = Dense(output_shape[-1])

                # data flow
                encoder_input = Input(input_shape)
                encoder_output, state_h = encoder(encoder_input)
                encoder_state = state_h  
                
                # initial input and state for iteration
                iter_input = Reshape((1,output_shape[-1]))(decoder_dense(state_h))
                iter_state = encoder_state
                all_output = []

                for _ in range(output_shape[0]):
                    # Run the decoder on one timestep, output == state_h since only one time step
                    output, state_h = decoder(iter_input, initial_state=iter_state)
                    output = decoder_dense(output)

                    # Store the current prediction (we will concatenate all predictions later)
                    all_output.append(output)
                    
                    # Reinject the outputs and state for the next loop iteration
                    iter_input = output
                    iter_state = state_h

            # Concatenate all predictions
            decoder_output = Lambda(lambda x: K.concatenate(x, axis=1))(all_output)
            self.model = Model(encoder_input, decoder_output)