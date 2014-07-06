data_dir=./data
n_dim=56354
n_label=2

all: train

train:
	./logistic_sgd.py -t --train $(data_dir)/train.svm --test $(data_dir)/test.svm --model $(data_dir)/model.npy -d $(n_dim) -l $(n_label)

predict:
	./logistic_sgd.py -p --model $(data_dir)/model.npy --test $(data_dir)/test.svm --pred $(data_dir)/pred.out -d $(n_dim) -l $(n_label)
