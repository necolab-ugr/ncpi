import os
import numpy as np
import h5py
import subprocess

class Features:
    '''
    Class for computing features from electrophysiological data recordings.
    '''
    def __init__(self):
        # class attributes come here
        pass

    def extract_features(self,method,data,config):
        '''
        # Call the feature extraction method
        :param method:
        :param data:
        :return:
        '''
        return method(data,config)

    def install(self,module):
        '''
        Function to install a Python module.

        Parameters
        ----------
        module: str
            Module name.
        '''
        subprocess.check_call(['pip', 'install', module])
        print(f"The module {module} was installed!")

    def hctsa(self,samples,hctsa_folder,workers = 32):
        '''
        Compute hctsa features.

        Parameters
        ----------
        samples: ndarray of shape (n_samples, times-series length)
            A set of samples of time-series data.
        hctsa_folder: str
            Folder where hctsa is installed.
        workers: int
            Number of MATLAB workers of the parallel pool.

        Returns
        -------
        feats: list of shape (n_samples, n_features)
            hctsa features.

        Debugging
        ---------
        This function has been debugged by approximating results shown
        in https://github.com/benfulcher/hctsaTutorial_BonnEEG.
        '''

        # import module
        try:
            import matlab.engine
        # We assume that Matlab and hctsa are installed
        except ImportError:
            print("MATLAB Engine is not installed!")
            self.install('matlabengine')
            import matlab.engine

        feats = []

        # remove hctsa file
        if os.path.isfile(os.path.join(hctsa_folder,'HCTSA.mat')):
            os.remove(os.path.join(hctsa_folder,'HCTSA.mat'))

        # start Matlab engine
        print("\n--> Starting Matlab engine ...")
        eng = matlab.engine.start_matlab()

        # change to hctsa folder
        eng.cd(hctsa_folder)

        # startup script
        print("\n--> hctsa startup ...")
        st = eng.startup(nargout=0)
        print(st)

        # create the input variables in Matlab
        eng.eval(f'timeSeriesData = cell(1,{samples.shape[0]});',nargout = 0)
        eng.eval(f'labels = cell(1,{samples.shape[0]});',nargout = 0)
        eng.eval(f'keywords = cell(1,{samples.shape[0]});',nargout = 0)

        # transfer time-series data to Matlab
        for s in range(samples.shape[0]):
            eng.workspace['aux'] = matlab.double(list(samples[s]))
            eng.eval('timeSeriesData{1,%s} = aux;' % (s+1),nargout = 0)

        # fill in the other 2 Matlab structures with the index of the sample
        for s in range(samples.shape[0]):
            eng.eval('labels{1,%s} = \'%s\';' % (str(s+1),str(s+1)),nargout = 0)
            eng.eval('keywords{1,%s} = \'%s\';' % (str(s+1),str(s+1)),nargout = 0)

        # Save variables into a mat file
        eng.eval('save INP_ccpi_ts.mat timeSeriesData labels keywords;',nargout = 0)

        # load mat file
        eng.eval('load INP_ccpi_ts.mat;',nargout = 0)

        # initialize an hctsa calculation
        print("\n--> hctsa TS_Init ...")
        eng.TS_Init('INP_ccpi_ts.mat',
                    'hctsa',
                    matlab.logical([False,False,False]),
                    nargout = 0)

        # open a parallel pool of a specific size
        if workers > 1:
            eng.parpool(workers)

        # compute features
        print("\n--> hctsa TS_Compute ...")
        # eng.TS_Compute(matlab.logical([True]),nargout = 0)
        eng.eval('TS_Compute(true);',nargout = 0)

        # load hctsa file
        f = h5py.File(os.path.join(hctsa_folder,'HCTSA.mat'),'r')
        TS_DataMat = np.array(f.get('TS_DataMat'))
        # TS_Quality = np.array(f.get('TS_Quality'))

        # create the array of features to return
        print(f'\n--> Formatting {TS_DataMat.shape[0]} features...')
        for s in range(samples.shape[0]):
            feats.append(list(TS_DataMat[:,s]))

        # stop Matlab engine
        print("\n--> Stopping Matlab engine ...")
        eng.quit()
        return feats

    def catch22(self,sample):
        '''
        Compute catch22 features.

        Parameters
        ----------
        sample: ndarray or list of shape (times-series length,)
            Time-series data sample.

        Returns
        -------
        feats: list of shape (22,)
            Values of the catch22 features.

        Debugging
        ---------
        This function has been debugged by replicating expected outputs
        included in benchmarks of the pycatch22 repository:
        https://github.com/DynamicsAndNeuralSystems/pycatch22/tree/main/tests.
        '''

        # import module
        try:
            import pycatch22
        except ImportError:
            print("pycatch22 is not installed!")
            self.install('pycatch22')

        feats = pycatch22.catch22_all(sample)
        return feats['values']