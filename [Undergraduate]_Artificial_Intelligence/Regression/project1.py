from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    '''
    Dataset class
    '''
    def __init__(self, data_path):
        '''
        Description:
            Load a csv file using 'load_data' method.
            Convert the dataframe to numpy array.
        Args:
            data_path (str): Path of the csv file.
        Returns:
        '''
        assert type(data_path)==str, 'The type of data_path has to be string'
        ### CODE HERE ###
        self.data_path = data_path                  # data_path 저장
        self.data = self.load_data()                # load_data 메소드를 이용하여 data 불러오기
        self.features, self.data_values = self.data # features와 data_values 저장
        # raise NotImplementedError
        #################
    
    def load_data(self):
        '''
        Description:
            Load the csv file.
            Print the head of data.
            Gather names of the features & values of the data.
        Args:
        Returns:
            features (list): The names of the features.
            data (numpy array): Data.
        '''
        ### CODE HERE ###
        df = pd.read_csv(self.data_path)    # Load the csv file
        print(df.head())                    # Show the first few rows of data
        features = list(df.columns)         # features 저장
        data = df.values                    # data 저장
        # raise NotImplementedError
        #################
        return features, data
    
    def gather_data(self,X,y,num_of_data):
        '''
        Description:
            Gather the first #(num_of_data) of the X,y dataset
        Args:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'ViolentCrimesPerPop')
            num_of_data (int): Number of data that should be sliced.
        Returns:
            X (numpy array): Sliced input data
            y (numpy arary): Sliced target 
        '''
        X=X[:num_of_data,:]
        y=y[:num_of_data]
        return X,y
    
    def parse_data(self, features):
        '''
        Description:
            Parse the data using 'features'
            Initialize bias 1 and concatenate it in front of input data. (10.12 update)
        Args:
            features (list): The names of feature we use.
        Returns:
            X (numpy array): Input data
            y (numpy arary): Target (feature name: 'ViolentCrimesPerPop')
        '''
        assert type(features)==list, 'The type of feature_names has to be list'
        assert all([isinstance(feature, str) for feature in features]), 'The element of features has to be string'
        
        ### CODE HERE ###
        df = pd.read_csv(self.data_path)                                # Load the csv file
        
        y = df['ViolentCrimesPerPop'].values                            # Separate the target data
        
        X = df[features].drop(columns=['ViolentCrimesPerPop']).values   # Select columns based on provided features & Remove the target column from the dataframe

        # Initialize bias of 1 and concatenate it in front of input data
        bias = np.ones((X.shape[0], 1))
        X = np.concatenate((bias, X), axis=1)

        # raise NotImplementedError
        #################
        return X, y


class Regressor:
    '''
    Regressor class
    '''
    def __init__(self, tau, dim):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        ### CODE HERE ###
        self.tau = tau                # 수렴 조건
        self.dim = dim                # feature의 개수
        # raise NotImplementedError
        #################
    
    def initialize_weight(self):
        '''
        Description:
            Initialize the weight of the model using normal distribution.
        Args:
        Returns:
            weight (numpy array): Initialized weight.
        '''
        np.random.seed(0)
        ### CODE HERE ###
        self.weight = np.random.normal(0, 1, size=self.dim)  # weight 초기화, 평균 0, 표준편차 1인 정규분포로 초기화
        return self.weight                                   # 초기화된 weight return
        # raise NotImplementedError
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        '''
        pass
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        '''
        pass
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        '''
        pass
    
    def loss_history(self):
        '''
        Description:
            Plot the history of the RSS loss.
        Args:
        Returns:
        '''
        ### CODE HERE ###
        plt.title('RSS Loss Over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('RSS loss')
        plt.plot(self.losses)        # Plot the loss history
        # plt.show()                   # 그래프 출력
        # raise NotImplementedError
        #################


class LinearRegressor(Regressor):
    '''
    Linear regressor class
    '''
    def __init__(self, tau, dim, lr=1e-5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lr)==float, 'The type of lr has to be float'
        ### CODE HERE ###
        super().__init__(tau, dim) # 상위 클래스의 __init__ 메소드 호출, tau와 dim을 받음
        self.lr = lr               # learning rate
        # raise NotImplementedError
        #################
        '''
        Description:
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 
            weight_coordinate_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.

            2. 정규화를 위한 사전 계산을 수행함:
            X 행렬의 각 열에 대해 제곱의 합을 계산하여 
            z 배열을 생성함.

            3. 반복을 위한 변수를 초기화함:
            초기 weight를 current_w에 저장함.

            4. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            - 4.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            - 4.2 residual을 계산함:
                    각 feature에 대해 해당 feature의 기여도를 제외한 
                    residual을 계산함.
            - 4.3 rho를 계산함:
                    residual과 현재 feature 값을 곱해 rho 값을 구함.
            - 4.4 weight를 업데이트함:
                    계산된 rho를 사용해 current_w의 해당 
                    weight를 업데이트함.

            5. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 
            기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함.

            6. 손실을 기록함:
            반복이 끝날 때마다 계산된 손실을 
            losses 리스트에 추가함.
        '''
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        ### CODE HERE ###
        # Initialize weight
        self.weight_coordinate_descent = self.initialize_weight()                       # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]                   # 초기 loss 계산 후 저장
        
        # Precompute z values for normalization
        z = np.sum(np.square(X), axis=0)                                                # normalizer (z)를 구함, z[j] = 행렬 X의 모든 행에서 열 j만 선택한 것의 제곱의 합

        # Initialize variables
        current_w = deepcopy(self.weight_coordinate_descent)                            # weight를 복사하여 current_w에 저장
        
        while True:
            prev_w = deepcopy(current_w)                                                # 이전 weight를 저장
            
            for j in range(self.dim):                                                   # 모든 feature에 대해 반복
                residual = y - np.dot(X, current_w) + np.dot(X[:, j], current_w[j])     # Residual vector (y - Xw) + X_j * w_j, Compute the residual without including feature j's contribution
                rho_j = np.dot(np.transpose(X[:, j]), residual)                         # rho 계산
                current_w[j] = rho_j / z[j]                                             # j번째 열의 weight를 업데이트 

            if max(np.absolute(prev_w - current_w)) < self.tau:                         # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                # print(max(np.absolute(prev_w - current_w)))
                # print(prev_w - current_w)
                self.weight_coordinate_descent = deepcopy(prev_w)                       # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X, y, current_w))                                # 현재 weight로 계산한 RSSloss를 losses list에 추가
        # print("losses: ", self.losses)
                    
            
        ############################## 이게 강의안에 존재하는 수렴조건이긴 한데, 이번 프로젝트는 이 것을 안 쓰네요 ##############################
        # 이것 때문에 시간 미친듯이 날렸네 ㅠㅠㅠㅠ                                                                                          #
        # delta_cost = -2 * np.dot(X.T, (y - np.dot(X, current_w)))                   # delta_cost 계산, -2 * X^T * (y - Xw)              #
        # if np.sum(np.absolute(delta_cost)) < self.tau:                              # delta_cost의 합이 tau보다 작으면 수렴               #
        #     self.weight_coordinate_descent = deepcopy(current_w)                    # 최종 weight 저장                                   #
        #     break                                                                   # 수렴하면 반복문 탈출                                #
        ###################################################################################################################################
        # raise NotImplementedError
        #################
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. X의 전치행렬을 계산함:
            np.transpose(X)를 사용하여 X의 전치행렬을 X_transpose에 저장함.

            2. weight를 계산함:
            closed-form solution을 사용하여 weight를 계산함.
            계산식은 w_hat = (X^T * X)^(-1) * X^T * y이며, 
            이를 통해 weight_closed_form에 결과를 저장함.
        '''

        ### CODE HERE ###
        # Closed-form solution for linear regression:
        # [w_hat = (X^T * X)^(-1) * X^T * y]

        X_transpose = np.transpose(X)                                                                    # X의 전치행렬
        self.weight_closed_form = np.dot(np.linalg.inv(np.dot(X_transpose, X)), np.dot(X_transpose, y))  # weight 계산 (w_hat = (X^T * X)^(-1) * X^T * y)
        # raise NotImplementedError
        #################
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 weight_gradient_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.
            초기 weight를 current_w에 복사하여 저장함.

            2. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            
            - 2.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            
            - 2.2 RSS 손실의 gradient를 계산함:
                    partial 변수에 -2 * np.dot(np.transpose(X), y - np.dot(X, prev_w))를 사용하여 
                    RSS loss의 gradient를 계산함.
            
            - 2.3 weight를 업데이트함:
                    계산된 gradient를 기반으로 weight를 업데이트하며, 
                    업데이트 식은 w^(t+1) = w^(t) - learning_rate * gradient임.

            3. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함. 최종 weight를 weight_gradient_descent에 저장함.

            4. 손실을 기록함:
            반복이 끝날 때마다 업데이트된 weight로 loss를 계산하고,
            losses 리스트에 저장함.
        '''

        ### CODE HERE ###
        self.weight_gradient_descent = self.initialize_weight()             # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_gradient_descent)]         # 초기 loss 계산 후 저장
        current_w = deepcopy(self.weight_gradient_descent)                  # weight를 복사하여 current_w에 저장
        
        while True:
            prev_w = deepcopy(current_w)                                    # 이전 weight를 저장
            partial = -2 * np.dot(np.transpose(X), y - np.dot(X, prev_w))   # RSS loss의 gradient 계산
            current_w = prev_w - self.lr * partial                          # weight 업데이트 (w^(t+1) = w^(t) - learning_rate * gradient)
            
            if max(np.absolute(prev_w - current_w)) < self.tau:             # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                self.weight_gradient_descent = deepcopy(prev_w)             # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X, y, current_w))                    # 업데이트된 weight로 loss를 계산 후 저장
        # raise NotImplementedError
        #################


class RidgeRegressor(Regressor):
    '''
    Ridge regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0, lr=1e-5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
            lr (float): Learning rate.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lam)==float or type(lam)==int, 'The type of lam has to be float or int'
        assert type(lr)==float, 'The type of lr has to be float'
        ### CODE HERE ###
        # Attributes
        self.dim = dim            # Number of features
        self.tau = tau            # Convergence threshold
        self.lam = lam            # Regularization parameter
        self.lr = lr              # Learning rate
        
        # Initialize weights for different learning methods
        # self.weight_coordinate_descent = None
        # self.weight_closed_form = None
        # self.weight_gradient_descent = None
        
        # raise NotImplementedError
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 weight_coordinate_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.

            2. 정규화를 위한 사전 계산을 수행함:
            X 행렬의 각 열에 대해 제곱의 합을 계산하여 z 배열을 생성함.
            (regularization을 위해 z 배열의 1번 인덱스 이후에 self.lam을 더할 수 있음.)

            3. 반복을 위한 변수를 초기화함:
            초기 weight를 current_w에 복사하여 저장함.

            4. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            
            - 4.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            
            - 4.2 각 feature에 대해 반복함:
                    각 feature에 대해 residual을 계산하고, rho_j를 계산함.
                    
                    - 4.2.1 feature이 intercept일 때 (j가 0일 때):
                        regularization 없이 current_w를 업데이트함.
                    
                    - 4.2.2 feature이 intercept가 아닐 때 (j가 0이 아닐 때):
                        regularization을 적용하여 current_w를 업데이트함.
            
            5. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함. 최종 weight를 weight_coordinate_descent에 저장함.

            6. 손실을 기록함:
            반복이 끝날 때마다 업데이트된 weight로 RSS 손실을 계산하고,
            losses 리스트에 추가함.
        '''

        ### CODE HERE ###
        self.weight_coordinate_descent = self.initialize_weight()                       # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]                   # 초기 loss 계산 후 저장
        # print('\t# weight_coordinate_decent: ',self.weight_coordinate_descent)
        # print('\t# losses: ',self.losses)

        z = np.sum(np.square(X), axis=0)                                                # normalizer (z)를 구함, z[j] = 행렬 X의 모든 행에서 열 j만 선택한 것의 제곱의 합
        # print("z: ", z)
        # z[1:] += self.lam                                                               # Add regularization only to non-bias weights (index 1 and beyond)
        # print("z_ridge: ", z)
        current_w = deepcopy(self.weight_coordinate_descent)                            # weight 복사하여 current_w에 저장

        # Start coordinate descent
        while True:                                                                     # 수렴할 때까지 반복
            prev_w = deepcopy(current_w) 
            for j in range(self.dim):                                                   # 모든 feature에 대해 반복
                residual = y - np.dot(X, current_w) + np.dot(X[:, j], current_w[j])     # Residual vector: (y - Xw) + X_j * w_j
                rho_j = np.dot(np.transpose(X[:, j]), residual)                         # rho_j 계산
                
                if j == 0:                                                              # j가 0일 때, 즉 intercept항일 때, regularization 없이 current_w를 업데이트
                    current_w[j] = rho_j / z[j]                                         
                else:                                                                   
                    current_w[j] = rho_j / (z[j] + self.lam)                            # j번째 열의 current_w를 업데이트, regularization 존재
            
            if max(np.absolute(prev_w - current_w)) < self.tau:                         # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                self.weight_coordinate_descent = deepcopy(prev_w)                       # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X,y,current_w))                                  # 현재 current_w로 계산한 RSSloss를 losses list에 추가
        # raise NotImplementedError
        #################
    
    def learning_with_closed_form(self, X, y):
        '''
        Description:
            Set attribute weight_closed_form using the closed form solution.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. 단위행렬을 생성함:
            X의 열 수에 맞는 크기의 단위행렬(identity_matrix)을 생성하고,
            첫 번째 원소를 0으로 설정하여 intercept 처리를 수행함.

            2. Regularization 행렬을 계산함:
            lambda 값과 단위행렬을 곱하여 regularization_matrix를 생성함.

            3. 역행렬 계산을 위한 항을 구함:
            (X^T * X + lambda * I)의 역행렬을 inverse_term에 저장함.

            4. weight를 계산함:
            closed-form solution을 사용하여 weight를 계산함.
            계산식은 w_hat = (X^T * X + lambda * I)^(-1) * X^T * y이며, 
            이를 통해 weight_closed_form에 결과를 저장함.

            5. 최종 손실을 계산하고 저장함:
            최종적으로 계산된 weight를 사용하여 RSS 손실을 계산하고,
            losses 리스트에 추가함.
        '''

        ### CODE HERE ###
        identity_matrix = np.identity(X.shape[1])                                           # 차원이 (X.shape[1], X.shape[1])인 단위행렬 생성
        identity_matrix[0,0] = 0                                                            # intercept 처리를 위해 첫 번째 element를 0으로 바꿈
        regularization_matrix = self.lam * identity_matrix                                  # Regularization term

        inverse_term = np.linalg.inv(np.dot(np.transpose(X), X) + regularization_matrix)    # (X^T * X + lambda * I)^(-1)
        self.weight_closed_form = np.dot(np.dot(inverse_term, np.transpose(X)), y)          # weight 계산, w_hat = (X^T * X + lambda * I)^(-1) * X^T * y
        self.losses.append(RSSloss(X, y, self.weight_closed_form))                          # 최종 loss 계산 후 저장
        # raise NotImplementedError
        #################
    
    def learning_with_gradient_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_gradient_descent using initialize_weight method.
            Update weight_gradient_descent until convergence using gradient descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 weight_gradient_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.
            초기 weight를 current_w에 복사하여 저장함.

            2. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            
            - 2.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            
            - 2.2 예측값을 계산함:
                    X와 current_w를 곱하여 y_pred를 계산함 (y_hat = X * weight).
            
            - 2.3 RSS 손실의 gradient를 계산함:
                    partial 변수에 -2 * np.dot(np.transpose(X), (y - y_pred))를 사용하여 
                    RSS 손실의 gradient를 계산함.
            
            - 2.4 weight를 업데이트함:
                    계산된 gradient와 regularization을 기반으로 weight를 업데이트하며,
                    업데이트 식은 w^(t+1) = (1 - 2 * lr * lam) * w^(t) - learning_rate * gradient임.

            3. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함. 최종 weight를 weight_gradient_descent에 저장함.

            4. 손실을 기록함:
            반복이 끝날 때마다 업데이트된 weight로 RSS 손실을 계산하고,
            losses 리스트에 추가함.
        '''

        ### CODE HERE ###
        self.weight_gradient_descent = self.initialize_weight()                            # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_gradient_descent)]                        # 초기 loss 계산 후 저장
        current_w = deepcopy(self.weight_gradient_descent)                                 # weight 복사하여 current_w에 저장
                     
        # Start gradient descent loop
        while True:
            prev_w = deepcopy(current_w)                                                   # 이전 weight를 저장
            y_pred = np.dot(X, current_w)                                                  # y_pred 계산, y_hat = X * weight
            partial = -2 * np.dot(np.transpose(X), (y - y_pred))                           # RSS loss의 gradient 계산
            current_w = (1 - 2 * self.lr * self.lam) * prev_w - self.lr * partial          # weight 업데이트, w^(t+1) = (1-2*lr*lamda)*w^(t) - learning_rate * gradient
        
            if max(np.absolute(prev_w - current_w)) < self.tau:                            # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                self.weight_gradient_descent = deepcopy(prev_w)                            # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X, y, current_w))                                   # Loss를 저장
        # raise NotImplementedError
        #################


class LassoRegressor(Regressor):
    '''
    Lasso regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
        Returns:
        '''
        assert type(tau)==float, 'The type of tau has to be float'
        assert type(dim)==int, 'The type of dim has to be integer'
        assert type(lam)==float or type(lam)==int, 'The type of lam has to be float or int'
        ### CODE HERE ###
        # 초기화
        self.dim = dim                              # feature의 개수
        self.tau = tau                              # 수렴 조건
        self.lam = lam                              # Regularization parameter
        self.weight = self.initialize_weight()      # 가중치를 0으로 초기화
        # raise NotImplementedError
        #################
    
    def thresholding(self, rho, lam):
        '''
        Description:
            Implement the thresholding function.
        Args:
            rho (float): The value to be thresholded.
            lam (float): Regularization parameter.
        Returns:
            out (float): The thresholded value.
        '''
        ### CODE HERE ###
        if rho < (-lam / 2):           # rho가 -lambda/2보다 작으면:                      rho + lambda/2 반환
            return (rho + lam / 2)    
        elif rho > (lam / 2):          # rho가 lambda/2보다 크면:                         rho - lambda/2 반환
            return (rho - lam / 2)    
        else:                          # 그 외의 경우, 즉 -lambda/2 <= rho <= lambda/2:   0 반환
            return 0                 
        # raise NotImplementedError
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 weight_coordinate_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.

            2. 정규화를 위한 사전 계산을 수행함:
            X 행렬의 각 열에 대해 제곱의 합을 계산하여 z 배열을 생성함.

            3. 반복을 위한 변수를 초기화함:
            초기 weight를 current_w에 복사하여 저장함.

            4. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            
            - 4.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            
            - 4.2 각 feature에 대해 반복함:
                    각 feature에 대해 residual을 계산하고, rho_j를 계산함 (X^T * residual).
                    
                    - 4.2.1 feature가 intercept일 때 (j가 0일 때):
                        regularization 없이 current_w를 업데이트함.
                    
                    - 4.2.2 feature가 intercept가 아닐 때 (j가 0이 아닐 때):
                        thresholding 메서드를 사용하여 regularization을 적용한 후
                        current_w를 업데이트함.
            
            5. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함. 최종 weight를 weight_coordinate_descent에 저장함.

            6. 손실을 기록함:
            반복이 끝날 때마다 업데이트된 weight로 RSS 손실을 계산하고,
            losses 리스트에 추가함.
        '''

        ### CODE HERE ###
        self.weight_coordinate_descent = self.initialize_weight()                       # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]                   # 초기 loss 계산 후 저장

        z = np.sum(np.square(X), axis=0)                                                # normalizer (z)를 구함, z[j] = 행렬 X의 모든 행에서 열 j만 선택한 것의 제곱의 합
        current_w = deepcopy(self.weight_coordinate_descent)                            # weight 복사하여 current_w에 저장

        # Start coordinate descent
        while True:                                                                     # 수렴할 때까지 반복
            prev_w = deepcopy(current_w)                                                # 이전 weight를 저장
            for j in range(self.dim):                                                   # 모든 feature에 대해 반복
                residual = y - np.dot(X, current_w) + np.dot(X[:, j], current_w[j])     # Residual vector: (y - Xw) + X_j * w_j
                rho_j = np.dot(np.transpose(X[:, j]), residual)                         # rho_j 계산, X^T * residual

                if j == 0:                                                              # j가 0일 때, 즉 intercept일 때
                    current_w[j] = rho_j / z[j]                                         # regularization 없이 current_w를 업데이트
                else:                                                                   # j가 0이 아닐 때, 즉 intercept가 아닐 때
                    current_w[j] = self.thresholding(rho_j, self.lam) / z[j]            # j번째 열의 current_w를 업데이트
            
            if max(np.absolute(prev_w - current_w)) < self.tau:                         # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                self.weight_coordinate_descent = deepcopy(prev_w)                       # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X,y,current_w))                                  # 현재 current_w로 계산한 RSSloss를 losses list에 추가
        # raise NotImplementedError
        #################


class ElasticNetRegressor(Regressor):
    '''
    ElasticNet regressor class
    '''
    def __init__(self, dim, tau=1e-3, lam=0, alpha=0.5):
        '''
        Description:
            Set inputs as attributes of this class.
        Args:
            dim (int): The number of features.
            tau (float): Threshold for convergence.
            lam (float or int): Regularization parameter.
            alpha (float): Elastic net mixing parameter (0 = ridge, 1 = lasso).
        Returns:
        '''
        assert type(tau) == float, 'The type of tau has to be float'
        assert type(dim) == int, 'The type of dim has to be integer'
        assert type(lam) == float or type(lam) == int, 'The type of lam has to be float or int'
        assert type(alpha) == float, 'The type of alpha has to be float'
        ### CODE HERE ###
        self.dim = dim                              # feature의 개수
        self.tau = tau                              # 수렴 조건
        self.lam = lam                              # Regularization parameter
        self.alpha = alpha                          # Elastic net mixing parameter
        self.weight = self.initialize_weight()      # weight 초기화  
        # raise NotImplementedError
        #################
    
    def thresholding(self, rho, lam):
        '''
        Description:
            Implement the thresholding function for Lasso regularization.
        Args:
            rho (float): The value to be thresholded.
            lam (float): Regularization parameter.
        Returns:
            out (float): The thresholded value.
        '''
        ### CODE HERE ###
        if rho < (-lam * self.alpha / 2):           # rho가 -lambda * alpha /2보다 작으면:                               rho + lambda/2 반환
            return (rho + lam * self.alpha / 2)    
        elif rho > (lam * self.alpha / 2):          # rho가 lambda * alpha /2보다 크면:                                  rho - lambda/2 반환
            return (rho - lam * self.alpha / 2)    
        else:                                       # 그 외의 경우, 즉 -lambda * alpha /2 <= rho <= lambda * alpha /2:   0 반환
            return 0      
        # raise NotImplementedError
        #################
    
    def learning_with_coordinate_descent(self, X, y):
        '''
        Description:
            Initialize the weight as weight_coordinate_descent using initialize_weight method.
            Update weight_coordinate_descent until convergence using coordinate descent method.
            Save list of losses over the number of iterations including the initial loss and the final loss.
        Args:
            X (numpy array): Input data.
            y (numpy array or float): Target data.
        Returns:
        '''
        '''
            1. weight를 초기화함:
            initialize_weight 메서드를 사용하여 weight_coordinate_descent를 초기화하고,
            초기 손실을 계산하여 losses 리스트에 저장함.

            2. 정규화를 위한 사전 계산을 수행함:
            X 행렬의 각 열에 대해 제곱의 합을 계산하여 z 배열을 생성함.

            3. 반복을 위한 변수를 초기화함:
            초기 weight를 current_w에 복사하여 저장함.

            4. 반복문을 시작함:
            weight가 수렴할 때까지 반복함.
            
            - 4.1 이전 weight를 저장함:
                    prev_w에 현재 weight를 저장함.
            
            - 4.2 각 feature에 대해 반복함:
                    각 feature에 대해 residual을 계산하고, rho_j를 계산함 (X^T * residual).
                    
                    - 4.2.1 feature가 intercept일 때 (j가 0일 때):
                        regularization 없이 current_w를 업데이트함.
                    
                    - 4.2.2 feature가 intercept가 아닐 때 (j가 0이 아닐 때):
                        thresholding 메서드를 사용하여 regularization을 적용한 후
                        current_w를 업데이트함. regularization 식은
                        (z[j] + self.lam * (1 - self.alpha))로 계산됨.
            
            5. 수렴을 검사함:
            이전 weight와 현재 weight의 절대 차이가 기준 tau 이하이면 수렴으로 판단하고 
            반복을 종료함. 최종 weight를 weight_coordinate_descent에 저장함.

            6. 손실을 기록함:
            반복이 끝날 때마다 업데이트된 weight로 RSS 손실을 계산하고,
            losses 리스트에 추가함.
        '''

        ### CODE HERE ###
        self.weight_coordinate_descent = self.initialize_weight()                                               # weight 초기화
        self.losses = [RSSloss(X, y, self.weight_coordinate_descent)]                                           # 초기 loss 계산 후 저장
        
        # normalizer (z)를 구함, z[j] = 행렬 X의 모든 행에서 열 j만 선택한 것의 제곱의 합
        z = np.sum(np.square(X), axis=0)                                                                              
        current_w = deepcopy(self.weight_coordinate_descent)                                                    # weight 복사하여 current_w에 저장

        # Start coordinate descent
        while True:
            prev_w = deepcopy(current_w) 
            for j in range(self.dim):                                                                           # 모든 feature에 대해 반복
                residual = y - np.dot(X, current_w) + np.dot(X[:, j], current_w[j])                             # Residual vector: (y - Xw) + X_j * w_j
                rho_j = np.dot(np.transpose(X[:, j]), residual)                                                 # rho_j 계산
                
                if j == 0:                                                                                      # j가 0일 때, 즉 intercept일 때
                    current_w[j] = rho_j / z[j]                                                                 # regularization 없이 current_w를 업데이트
                else:
                    current_w[j] = self.thresholding(rho_j, self.lam) / (z[j] + self.lam * (1-self.alpha))      # j번째 열의 current_w를 업데이트
            
            if max(np.absolute(prev_w - current_w)) < self.tau:                                                 # 이전 weight와 현재 weight의 차이가 tau보다 작으면 수렴
                self.weight_coordinate_descent = deepcopy(prev_w)                                               # 최종 weight 저장
                break
            
            self.losses.append(RSSloss(X, y, current_w))                                                        # 현재 current_w로 계산한 RSSloss를 losses list에 추가
        # raise NotImplementedError
        #################


def RSSloss(X, y, weight):
    '''
    Description:
        Calculate the RSS loss (error).
    Args:
        X (numpy array): Input data
        y (numpy array or float): Target data
        weight (numpy array): Weight of the model
    Returns:
        loss (float): RSS loss (error).
    '''
    ### CODE HERE ###
    residuals = y - np.dot(X, weight)   # Residuals 계산
    loss = np.sum(np.square(residuals)) # RSS loss 계산
    return loss                         # RSS loss 반환
    # raise NotImplementedError
    #################