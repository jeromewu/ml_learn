data_dir=../data
train_file_name=$(data_dir)/a1a.dat
test_file_name=$(data_dir)/a1a.t.dat
log_file_name=./args.log
n_folds=3

all: sel_svmarg_in

sel_svmarg_in:
	./sel_svmarg.py --train $(train_file_name) -v $(n_folds)

sel_svmarg_out:
	./sel_svmarg.py --train $(train_file_name) --test $(test_file_name)
