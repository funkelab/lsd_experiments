import os

setup_dir = os.path.dirname(os.path.realpath(__file__))

print(setup_dir)

exp_dir = os.path.join(setup_dir, '..', '..', '../cremi')

print(exp_dir)

auto_setup = os.path.realpath(os.path.join(
    exp_dir,
    '02_train'))

print(auto_setup)
