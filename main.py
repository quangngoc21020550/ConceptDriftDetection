import torch
import numpy as np

from EmbeddingNet import EmbeddingNet, cosine_similarity
import torch.optim as optim

from TrainCentroidMatrixNet import FineTunedNet
from preprocessing import LoadDriftData
from config import args
from torch.utils.data import TensorDataset,DataLoader
from prototypical_batch_sampler import PrototypicalBatchSampler
import json
import os
from torcheval.metrics.functional import multiclass_confusion_matrix

np.random.seed(0)
N_CLASSES = 4
N_LAST_ELEMENTS = 5
BATCH_SIZE = 40
DTYPE_MAP = {
    0: 'sudden',
    1: 'gradual',
    2: 'incremental',
    3: 'normal'
}

def init_dataset():
    # Reading the data
    print('Reading the train data')
    all_data_frame = LoadDriftData(args.Data_Vector_Length, args.DATA_FILE,args.DATA_SAMPLE_NUM)
    Drift_data_array = all_data_frame.values
    where_are_nan = np.isnan(Drift_data_array)
    where_are_inf = np.isinf(Drift_data_array)
    Drift_data_array[where_are_nan] = 0.0
    Drift_data_array[where_are_inf] = 0.0
    print(True in np.isnan(Drift_data_array))

    # random shuffle datasets
    np.random.shuffle(Drift_data_array)
    data_count = Drift_data_array.shape[0]  # data count
    train_x = Drift_data_array[0:int(data_count * args.Train_Ratio), 0:args.Data_Vector_Length]
    train_y = Drift_data_array[0:int(data_count * args.Train_Ratio), -2]
    train_locy = Drift_data_array[0:int(data_count * args.Train_Ratio), -1]

    finetune_x = Drift_data_array[int(data_count * args.Train_Ratio):int(data_count * (args.Train_Ratio + args.FineTune_Ratio)), 0:args.Data_Vector_Length]
    finetune_y = Drift_data_array[int(data_count * args.Train_Ratio):int(data_count * (args.Train_Ratio + args.FineTune_Ratio)), -2]
    finetune_locy = Drift_data_array[int(data_count * args.Train_Ratio):int(data_count * (args.Train_Ratio + args.FineTune_Ratio)), -1]

    test_x = Drift_data_array[int(data_count * (args.Train_Ratio + args.FineTune_Ratio)):, 0:args.Data_Vector_Length]
    test_y = Drift_data_array[int(data_count * (args.Train_Ratio + args.FineTune_Ratio)):, -2]
    test_locy = Drift_data_array[int(data_count * (args.Train_Ratio + args.FineTune_Ratio)):, -1]

    y=np.hstack((train_y,finetune_y,test_y))
    n_classes = len(np.unique(y))
    if n_classes < args.Nc:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return train_x,train_y,train_locy,test_x,test_y,test_locy,finetune_x, finetune_y,finetune_locy


def init_lr_scheduler(optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=args.lr_scheduler_gamma,
                                           step_size=args.lr_scheduler_step)


def init_sampler(labels):
    classes_per_it = args.Nc
    num_samples = args.Ns + args.Nq

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=args.iterations)


def init_dataloader():
    print("Init dataset")
    train_x,train_y,train_locy,test_x,test_y,test_locy, finetune_x, finetune_y,finetune_locy = init_dataset()
    sampler = init_sampler(train_y)
    # TODO 生成train DataLoader
    Train_DS = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y), torch.unsqueeze(torch.FloatTensor(train_locy),1))
    train_dataloader = DataLoader(Train_DS, batch_sampler=sampler)
    # TODO 生成test DataLoader
    sampler = init_sampler(test_y)
    Test_DS = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y), torch.unsqueeze(torch.FloatTensor(test_locy),1))
    test_dataloader = DataLoader(Test_DS, batch_sampler=sampler)

    sampler = init_sampler(finetune_y)
    # TODO FineTune DataLoader
    FineTune_DS = TensorDataset(torch.FloatTensor(finetune_x), torch.LongTensor(finetune_y),
                             torch.unsqueeze(torch.FloatTensor(finetune_locy), 1))
    finetune_dataloader = DataLoader(FineTune_DS, batch_sampler=sampler)
    return train_dataloader,test_dataloader, finetune_dataloader

def split_drift_type(dataloader: DataLoader):
    # sudden_data = []
    # gradual_data = []
    # incremental_data = []
    # normal_data = []
    data_dict = {
        0: torch.Tensor(),
        1: torch.Tensor(),
        2: torch.Tensor(),
        3: torch.Tensor()
    }
    # print(len(enumerate(dataloader)))
    for batch_idx, data in enumerate(dataloader):
        datax, datay, locy = data
        datay = datay.tolist()
        for i in range(len(datay)):
            xdata = torch.reshape(datax[i], (1,args.Data_Vector_Length))
            data_dict[datay[i]] = torch.cat((data_dict[datay[i]], xdata), dim=0)

    return list(data_dict.values())

train_dataloader, test_dataloader, finetune_dataloader = init_dataloader()
splitted_test_set = split_drift_type(test_dataloader)


def pretrain():

    ModelSelect = args.FAN # 'RNN', 'FAN', 'FNN','FQN','CNN'
    # train_dataloader,test_dataloader=init_dataloader()
    # splitted_test_set = split_drift_type(test_dataloader)
    print('Checking if GPU is available')
    use_gpu = torch.cuda.is_available()
    use_gpu = False

    # Set training iterations and display period
    num_episode = args.num_episode

    # Initializing prototypical net
    print('Initializing PreTrain net')
    def train():
        model = EmbeddingNet(use_gpu=use_gpu, Data_Vector_Length =args.Data_Vector_Length, ModelSelect=ModelSelect)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = init_lr_scheduler(optimizer)
        # loss_fn = torch.nn.CrossEntropyLoss()
        train_loss = []
        train_class_acc = []


        centroid_matrix=torch.Tensor()
        # Training loop
        for i in range(num_episode):
            model.train()
            for batch_idx,data in enumerate(train_dataloader):
                optimizer.zero_grad()
                datax, datay, _ = data

                loss, class_acc, centroid_matrix = model(datax, datay)

                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_class_acc.append(class_acc.item())
                centroid_matrix = centroid_matrix

            avg_loss = np.mean(train_loss)
            avg_class_acc = np.mean(train_class_acc)
            print('{} episode,Avg Train Loss: {}, Avg Train Class Acc: {}'.format(
                i,avg_loss, avg_class_acc))

            lr_scheduler.step()

        # Check whether the file exists
        folder_path = "./input/Model/"+args.DATA_FILE
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        PATH = './input/Model/'+args.DATA_FILE+'/{name}_model_embeding.pkl'.format(name=ModelSelect)
        torch.save(model.state_dict(), PATH)
        # save centroid matrix
        torch.save(centroid_matrix, './input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
        # save loss
        with open('./input/Model/' + args.DATA_FILE + '/{name}_pretrain_loss.json'.format(name=ModelSelect), "w") as f:
            json.dump({"train_loss":np.array(train_loss[len(train_loss)-5:]).sum()/len(train_loss[len(train_loss)-5:]),
                       "train_class_acc":np.array(train_class_acc[len(train_class_acc)-5:]).sum()/len(train_class_acc[len(train_class_acc)-5:])}, f)

    # train()

    # Test loop
    centroid_matrix = torch.load(
        './input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
    model = EmbeddingNet(use_gpu=use_gpu, Data_Vector_Length=args.Data_Vector_Length,
                                             ModelSelect=ModelSelect)
    PATH = './input/Model/' + args.DATA_FILE + '/' + ModelSelect + '_model_embeding.pkl'
    model.load_state_dict(torch.load(PATH))

    CM = torch.zeros(N_CLASSES,N_CLASSES)

    test_class_acc = []
    # time1 = time.time()
    for batch_idx, data in enumerate(test_dataloader):
        datax,datay,_ = data
        embedded_x = model.PrototypicalNet(datax)
        y_pred = torch.t(cosine_similarity(centroid_matrix, embedded_x))
        # y_pred = model.forward_test(datax, centroid_matrix)
        pred_labels = torch.argmax(y_pred, dim=1)
        class_acc = (pred_labels == datay).sum().item() / len(datay)
        test_class_acc.append(class_acc)
        CM = torch.add(CM, multiclass_confusion_matrix(pred_labels, datay, num_classes=N_CLASSES))

    # time2 = time.time()
    # print(time2 - time1)
    avg_class_acc = np.mean(test_class_acc)
    # avg_loc_acc = np.mean(test_loc_acc)
    # save result
    CM = CM.type(dtype=torch.int)
    print(f'Confusion Matrix Pretrain:\n {CM}')
    with open('./input/Model/' + args.DATA_FILE + '/{name}_pretrain_result.json'.format(name=ModelSelect), "w") as f:
        json.dump({"avg_class_acc":avg_class_acc}, f)

    print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
        avg_class_acc, ""))

def finetunning():
    ModelSelect = args.FAN  # 'RNN', 'FAN','CNN', 'FNN','FQN','FAN'
    # train_dataloader, test_dataloader = init_dataloader()
    # splitted_test_set = split_drift_type(test_dataloader)
    print('Checking if GPU is available')
    use_gpu = torch.cuda.is_available()
    use_gpu = False

    # Set training iterations and display period
    num_episode = args.finetune_num_episode
    # load teacher model
    BASE_PATH = './input/Model/' + args.DATA_FILE
    teacher_model = EmbeddingNet(use_gpu=use_gpu, Data_Vector_Length=args.Data_Vector_Length,
                                             ModelSelect=ModelSelect)
    PATH = BASE_PATH + '/' + ModelSelect + '_model_embeding.pkl'
    teacher_model.load_state_dict(torch.load(PATH))
    centroid_matrix = torch.load('./input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
    embedding_model = teacher_model.PrototypicalNet
    finetuning_model = FineTunedNet(centroid_matrix=centroid_matrix)
    # Initializing prototypical net
    def train():
        print('Initializing Finetunning net')
        # sub_model = Sub_Joint_Prediction(Data_Vector_Length=args.Data_Vector_Length)

        optimizer = optim.Adam(finetuning_model.parameters(), lr=0.05)
        lr_scheduler = init_lr_scheduler(optimizer)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        train_loss = []
        train_class_acc = []
        # train_loc_acc = []
        test_loss = []
        test_class_acc = []
        # test_loc_acc = []
        # Training loop
        for i in range(num_episode):
            finetuning_model.train()
            for batch_idx, data in enumerate(finetune_dataloader):
                optimizer.zero_grad()
                datax, datay, locy = data
                embedded_x = embedding_model(datax)
                y_pred = finetuning_model(embedded_x)
                loss = loss_fn(y_pred, datay)
                pred_labels = torch.argmax(y_pred, dim=1)
                class_acc = (pred_labels == datay).sum().item()/len(datay)
                train_class_acc.append(class_acc)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            avg_loss = np.mean(train_loss)
            avg_class_acc = np.mean(train_class_acc)
            print('{} episode,Avg Train Loss: {},Avg Train Class Acc: {}'.format(
                i, avg_loss, avg_class_acc))

            lr_scheduler.step()

        PATH = './input/Model/' + args.DATA_FILE + '/{name}_finetune_model_embeding.pkl'.format(name=ModelSelect)
        torch.save(finetuning_model.state_dict(), PATH)
        # save loss
        with open('./input/Model/' + args.DATA_FILE + '/{name}_finetune_loss.json'.format(name=ModelSelect),
                  "w") as f:
            json.dump({"train_loss": np.array(train_loss[-N_LAST_ELEMENTS:]).sum()/N_LAST_ELEMENTS, "train_class_acc": np.array(train_class_acc[-N_LAST_ELEMENTS:]).sum()/N_LAST_ELEMENTS}, f)
    # Test loop
    # train()

    finetuning_model = FineTunedNet(centroid_matrix=centroid_matrix)
    PATH = './input/Model/' + args.DATA_FILE + '/' + ModelSelect + '_finetune_model_embeding.pkl'
    finetuning_model.load_state_dict(torch.load(PATH))
    test_class_acc = []
    CM = torch.zeros(N_CLASSES, N_CLASSES)
    # time1 = time.time()
    for batch_idx, data in enumerate(test_dataloader):
        datax, datay, locy = data
        # type_pred_T, loc_pred_T, loc_W = teather_model(datax, BASE_PATH)
        # loss, type_loss, loc_loss, loc_acc, type_acc = sub_model(datax, datay, locy, type_pred_T, loc_pred_T, loc_W)
        embedded_x = embedding_model(datax)
        y_pred = finetuning_model(embedded_x)
        # loss = loss_fn(y_pred, datay)
        # test_loss.append(loss.item())
        pred_labels = torch.argmax(y_pred, dim=1)
        class_acc = (pred_labels == datay).sum().item() / len(datay)
        test_class_acc.append(class_acc)
        CM = torch.add(CM, multiclass_confusion_matrix(pred_labels, datay, num_classes=N_CLASSES))

    # time2 = time.time()
    # print(time2 - time1)
    CM = CM.type(dtype=torch.int)
    print(f'\nConfusion Matrix FineTune:\n {CM}')
    avg_class_acc = np.mean(test_class_acc)
    # avg_loc_acc = np.mean(test_loc_acc)
    # save result
    with open('./input/Model/' + args.DATA_FILE + '/{name}_finetune_result.json'.format(name=ModelSelect),
              "w") as f:
        json.dump({"avg_class_acc": avg_class_acc}, f)

    print(' Avg Test Class Acc: {}'.format(
        avg_class_acc))



if __name__ == "__main__":
    # TODO Pretrain
    pretrain()
    # TODO Finetune
    finetunning()

