from proj1.base import Base
class DR_NN():
    def run_nn(self, X_train_full, X_test, y_train_full, y_test, type, optimal_hidden_size=None):
        nn_Base = Base(X_train_full=X_train_full, X_test=X_test, y_train_full=y_train_full, y_test=y_test, scoring=['accuracy'],
                       type=type, scale= False, cv = 5, sub_Train_Set_Ratio_Start=1, optimal_hidden_size=optimal_hidden_size)
        score, fit_time = nn_Base.run()
        print(type)
        print('score: ' + str(score) + 'fit_time: ' + str(fit_time))