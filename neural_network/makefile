data_dir=../data
data=a1a
train_file_name=$(data_dir)/$(data).dat
test_file_name=$(data_dir)/$(data).t.dat
model_file_name=$(data_dir)/$(data).model.npy
mlp_model_file_name=$(data_dir)/$(data).mlp.model.npy
pred_file_name=$(data_dir)/$(data).pred.out
mlp_pred_file_name=$(data_dir)/$(data).mlp.pred.out
n_hidden_units=123
n_dim=123
n_label=10
learning_rate=0.01

all: sgdtrain

sgdtrain:
	./logistic_sgd.py -t --train $(train_file_name) --test $(test_file_name) --model $(model_file_name) -f $(n_dim) -l $(n_label) -r $(learning_rate)

sgdpredict:
	./logistic_sgd.py -p --model $(model_file_name) --test $(test_file_name) --pred $(pred_file_name) -f $(n_dim) -l $(n_label)

mlptrain:
	./mlp.py -t --train $(train_file_name) --test $(test_file_name) --model $(mlp_model_file_name) -f $(n_dim) -l $(n_label) --hunits $(n_hidden_units)

mlppredict:
	./mlp.py -p --model $(mlp_model_file_name) --test $(test_file_name) --pred $(mlp_pred_file_name) -f $(n_dim) -l $(n_label)

clean:
	rm -rf *.pyc ../data/*.npy
