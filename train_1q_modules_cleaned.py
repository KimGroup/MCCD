# python train_1q_modules_cleaned.py --if_final_round_syndrome --batch_size 1024 --logical_circuit_index 3
from mccd import MultiDepthCachedSyndromeDataset, CircuitLSTMDecoder
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os


def validate(model, loss_fn, dataloader, device):
    cumulative_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(dataloader):
        syndromes = data['syndromes'].squeeze(0).float().to(device)
        labels = data['label'].long().squeeze(0).to(device)
        final_round_syndromes = data['final_round_syndromes'].squeeze(0).float().to(device)
        circuit = data['circuit']

        outputs = model(syndromes, circuit, final_round_syndromes)
        loss = loss_fn(outputs[0], outputs[1], labels)

        outputs = outputs[0]
        _, predicted = torch.max(outputs.data, -1)

        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()
        cumulative_loss += loss.item()

    return 100 * correct / total, cumulative_loss / total


def get_collate_function(if_final_round_syndrome=False):
    return lambda batch: collate_fn(batch, if_final_round_syndrome)


def collate_fn(batch, if_final_round_syndrome=False):
    syndromes = torch.cat([b['syndromes'].unsqueeze(0) for b in batch])
    label = torch.cat([b['label'].unsqueeze(0) for b in batch])
    circuit = batch[0]['circuit']

    if if_final_round_syndrome:
        final_round_syndromes = torch.cat(
            [b['final_round_syndromes'].unsqueeze(0) for b in batch])
        return {
            'syndromes': syndromes,
            'label': label,
            'circuit': circuit,
            'final_round_syndromes': final_round_syndromes
        }

    return {
        'syndromes': syndromes,
        'label': label,
        'circuit': circuit
    }


if __name__ == '__main__':
    from parseargs import generate_save_path, parse_all_args

    args = parse_all_args()
    print(args, flush=True)

    code_distance = args.code_distance
    logical_circuit_index = args.logical_circuit_index
    assert logical_circuit_index == '3'  # Require Type I circuit

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_ancillas = int(code_distance ** 2 - 1)

    model = CircuitLSTMDecoder(num_ancillas, num_ancillas * 8, num_layers=2).to(device)

    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of params in model: {params}')

    save_path = os.path.join('trained_models', generate_save_path(None, mle=False))
    print(f'Save to {save_path}')
    os.makedirs(save_path, exist_ok=True)

    def loss_fn(main_out, auxiliary_out, label, weight=0.5):
        main_out = main_out.view(-1, main_out.size(-1))
        auxiliary_out = auxiliary_out.view(-1, auxiliary_out.size(-1))
        label = label.view(-1)
        H_main = nn.CrossEntropyLoss()(main_out, label)
        H_aux = nn.CrossEntropyLoss()(auxiliary_out, label)
        return H_main + weight * H_aux

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_accuracies = []
    testing_accuracies = []
    losses = []

    val_every_n_batches = args.validate_every_n_batches
    save_model_every_n_batches = args.save_model_every_n_batches
    
    cf = get_collate_function(if_final_round_syndrome=args.if_final_round_syndrome)

    train_dataset = MultiDepthCachedSyndromeDataset(
        root_dir=args.train_data_dir,
        code_distance=code_distance,
        circuit_index=logical_circuit_index,
        batch_size=args.batch_size,
        depth_list=args.train_depth_list
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=cf)

    val_dataset = MultiDepthCachedSyndromeDataset(
        root_dir=args.val_data_dir,
        code_distance=code_distance,
        circuit_index=logical_circuit_index,
        batch_size=args.batch_size,
        depth_list=args.val_depth_list
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=cf)
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    seen_samples = 0

    for i, data in tqdm(enumerate(train_dataloader)):
        syndromes = data['syndromes'].squeeze(0).float().to(device)
        labels = data['label'].squeeze(0).to(device)
        circuit = data['circuit']

        if args.if_final_round_syndrome:
            final_round_syndromes = data['final_round_syndromes'].squeeze(0).float().to(device)
        else:
            final_round_syndromes = None

        optimizer.zero_grad()
        outputs = model(syndromes, circuit, final_round_syndromes)
        loss = loss_fn(outputs[0], outputs[1], labels)
        loss.backward()
        optimizer.step()

        outputs = outputs[0]
        _, predicted = torch.max(outputs.data, -1)

        total += labels.size(0) * labels.size(1)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        seen_samples += args.batch_size

        if i % val_every_n_batches == 0:
            loss_val = running_loss / total
            losses.append(loss_val)
            training_acc = 100 * correct / total
            training_accuracies.append(training_acc)

            testing_acc, testing_loss = validate(model, loss_fn, val_dataloader, device)
            testing_accuracies.append(testing_acc)

            print(f"[Step {seen_samples}] Train Acc: {training_acc:.2f}% | Val Acc: {testing_acc:.2f}%")

        if i% save_model_every_n_batches == 0:
            model_path = os.path.join(save_path, f'model_{args.run_index}.pt')
            torch.save(model, model_path)
            print('Model checkpoint saved.', flush=True)

            running_loss = 0.0
            correct = 0.0
            total = 0.0
        
    model_path = os.path.join(save_path, f'model_{args.run_index}.pt')
    torch.save(model, model_path)
    print('Final model saved.', flush=True)
