class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'pajinsen':
            # folder that contains class labels
            root_dir = 'D:\my_Ai_project\hand-gesture-recognition-using-mediapipe-main\data_process\Pajinsen'

            # Save preprocess data into output_dir
            output_dir = 'D:\my_Ai_project\hand-gesture-recognition-using-mediapipe-main\data_process\Pajinsen'
            return root_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './model/c3d-pretrained.pth'