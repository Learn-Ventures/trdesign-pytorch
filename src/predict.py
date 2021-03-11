import fire, os
from tr_Rosetta_model import *

# cli function for ensemble prediction with pre-trained network
#@torch.no_grad()
def get_ensembled_predictions(input_file, output_file=None, model_dir='models/trRosetta_models'):
    structure_model = trRosettaNetwork()
    input_data      = preprocess(msa_file = input_file)
    #input_data      = preprocess(use_random_seq = True)

    if output_file is None:
        input_path  = Path(input_file)
        output_file = f'{input_path.parents[0] / input_path.stem}.npz'

    outputs = []
    for model_file in load_models(model_dir):
        structure_model.load_state_dict(torch.load(model_file, map_location=torch.device(d())))
        structure_model.to(d()).eval()
        output = structure_model(input_data) #prob_theta, prob_phi, prob_distance, prob_omega
        outputs.append(output)

    averaged_outputs = [torch.stack(model_output).mean(dim=0).cpu().detach().numpy() for model_output in zip(*outputs)]
    output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], averaged_outputs))
    np.savez_compressed(output_file, **output_dict)
    print(f'predictions for {input_file} saved to {output_file}')

    plot_distogram(distogram_distribution_to_distogram(output_dict['dist']), '%s_dist.jpg' %input_file)

'''
cd ...
python predict.py data/test.a3m
or 
python predict.py data/test.fasta
'''

if __name__ == '__main__':
    fire.Fire(get_ensembled_predictions)