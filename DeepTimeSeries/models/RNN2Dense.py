import numpy as np
from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Dropout, Reshape, Lambda
from keras import backend as K
from DeepTimeSeries.models.TimeSeriesBase import TSBase

class RNN2Dense_1(TSBase):
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
            self.class_info={'class': 'RNN2Dense_1', 'input_shape': input_shape, 
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



class RNN2Dense_2(TSBase):
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
            self.class_info={'class': 'RNN2Dense_2', 'input_shape': input_shape, 
                'output_shape': output_shape, 'cell': cell, 'cell_units': cell_units,
                'dense_units': dense_units} 

            # check input variables 
            assert cell in ['SimpleRNN', 'LSTM', 'GRU']
            assert type(cell_units) == int
            assert type(dense_units) == tuple
            
            # batch_input_shape = [n_sample, n_backtrack_step, n_feature] 
            x_in = Input(input_shape)
            if cell == 'SimpleRNN':
                x = SimpleRNN(units=cell_units, return_sequences=True)(x_in)
            elif cell == 'LSTM':
                x = LSTM(units=cell_units, return_sequences=True)(x_in)
            elif cell == 'GRU':
                x = GRU(units=cell_units, return_sequences=True)(x_in)
            # if dense_units != None:
            #     for n_units in dense_units:
            #         x = Dense(n_units, activation='relu')(x)
            #         x = Dropout(dropout_rate)(x)
            #x = Reshape((-1,output_shape[-1]*cell_units))(x)
            #x_out = Dense(20)(x)
            #x_out = Dense(np.prod(output_shape))(x)
            #x_out = Reshape((output_shape))(x)
            self.model = Model(inputs = x_in, outputs = x)