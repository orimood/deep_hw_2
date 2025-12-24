import json
import yaml
from pathlib import Path
import pandas as pd

# Load data from all wandb runs
wandb_dir = Path('results/wandb')
runs_data = []

for run_dir in sorted(wandb_dir.iterdir()):
    if run_dir.is_dir() and run_dir.name.startswith('run-'):
        summary_file = run_dir / 'files' / 'wandb-summary.json'
        config_file = run_dir / 'files' / 'config.yaml'
        
        if summary_file.exists() and config_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract date/time from run_id
            run_datetime = run_dir.name.split('-')[1]
            
            # Extract key metrics and architecture details
            run_info = {
                'run_datetime': run_datetime,
                'run_id': run_dir.name,
                'test_acc': summary.get('test_acc', summary.get('summary/test_acc')),
                'test_loss': summary.get('test_loss', summary.get('summary/test_loss')),
                'val_acc': summary.get('val_acc'),
                'val_loss': summary.get('val_loss'),
                'train_acc': summary.get('train_acc'),
                'train_loss': summary.get('train_loss'),
                'best_val_acc': summary.get('best_val_acc', summary.get('summary/best_val_acc')),
                'best_epoch': summary.get('best_epoch'),
                'runtime': summary.get('_runtime', summary.get('_wandb', {}).get('runtime')),
                'learning_rate': config.get('learning_rate', {}).get('value'),
                'batch_size': config.get('batch_size', {}).get('value'),
                'dropout': config.get('dropout', {}).get('value'),
                'weight_decay': config.get('weight_decay', {}).get('value'),
                'optimizer': config.get('optimizer', {}).get('value'),
                'architecture': config.get('architecture', {}).get('value'),
                'improvements': config.get('improvements', {}).get('value', ''),
                'loss_function': config.get('loss', {}).get('value'),
                'embedding_dim': config.get('embedding_dim', {}).get('value'),
                'scheduler': config.get('scheduler', {}).get('value'),
                'step_size': config.get('step_size', {}).get('value'),
                'gamma': config.get('gamma', {}).get('value'),
                'similarity_channels': config.get('similarity_channels', {}).get('value'),
                'epochs': config.get('epochs', {}).get('value'),
                '_step': summary.get('_step', 0)  # Last step/epoch completed
            }
            runs_data.append(run_info)

# Create DataFrame
df = pd.DataFrame(runs_data)
df = df.sort_values('run_datetime').reset_index(drop=True)

# Add run number
df.insert(0, 'run_num', range(1, len(df) + 1))

# Determine completion status
df['completed'] = df['_step'].apply(lambda x: 'Yes' if x >= 19 else 'No' if pd.notna(x) else 'Unknown')

# Calculate actual epochs completed
df['epochs_completed'] = df['_step'] + 1  # _step is 0-indexed

print("=" * 150)
print("ALL WANDB RUNS - DETAILED TABLE")
print("=" * 150)
print()

# Display full table
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print(df.to_string(index=False))

print("\n" + "=" * 150)
print("MARKDOWN TABLE FOR REPORT")
print("=" * 150)
print()

# Create markdown table with all relevant columns
columns_for_table = [
    'run_num', 'run_datetime', 'test_acc', 'val_acc', 'best_val_acc', 
    'test_loss', 'epochs_completed', 'completed',
    'learning_rate', 'batch_size', 'weight_decay', 'dropout',
    'embedding_dim', 'architecture', 'improvements'
]

# Only include columns that exist
columns_for_table = [col for col in columns_for_table if col in df.columns]

# Create summary of changes
print("\n### All Training Runs Summary\n")
print("This table shows all experimental runs conducted during the project, including incomplete runs:\n")

# Markdown table header
headers = ['Run', 'Date/Time', 'Test Acc (%)', 'Val Acc (%)', 'Best Val Acc (%)', 'Test Loss', 
           'Epochs', 'Completed', 'LR', 'Batch', 'Weight Decay', 'Dropout', 'Emb Dim', 'Key Changes']
print('| ' + ' | '.join(headers) + ' |')
print('|' + '|'.join(['---' for _ in headers]) + '|')

for idx, row in df.iterrows():
    # Determine key changes
    key_changes = []
    improvements = str(row.get('improvements', ''))
    
    if 'BatchNorm' in improvements:
        key_changes.append('BatchNorm')
    if 'Multi-channel' in improvements or 'Multi-Channel' in improvements:
        key_changes.append('Multi-channel')
    if row.get('weight_decay') == 0.001:
        key_changes.append('Higher WD')
    if row.get('weight_decay') == 0.0005:
        key_changes.append('WD=0.0005')
    if 'BCEWithLogits' in str(row.get('loss_function', '')):
        key_changes.append('BCEWithLogits')
    if 'Improved' in str(row.get('architecture', '')):
        key_changes.append('Improved Arch')
    if pd.notna(row.get('dropout')) and 'embedding' in str(row.get('dropout')).lower():
        key_changes.append('Higher Dropout')
    
    if not key_changes:
        key_changes.append('Baseline')
    
    # Format values
    test_acc = f"{row['test_acc']:.1f}" if pd.notna(row.get('test_acc')) else 'N/A'
    val_acc = f"{row['val_acc']:.1f}" if pd.notna(row.get('val_acc')) else 'N/A'
    best_val_acc = f"{row['best_val_acc']:.1f}" if pd.notna(row.get('best_val_acc')) else 'N/A'
    test_loss = f"{row['test_loss']:.4f}" if pd.notna(row.get('test_loss')) else 'N/A'
    epochs = f"{int(row['epochs_completed'])}/{int(row['epochs'])}" if pd.notna(row.get('epochs_completed')) and pd.notna(row.get('epochs')) else 'N/A'
    completed = row.get('completed', 'Unknown')
    lr = f"{row['learning_rate']}" if pd.notna(row.get('learning_rate')) else 'N/A'
    batch = f"{int(row['batch_size'])}" if pd.notna(row.get('batch_size')) else 'N/A'
    wd = f"{row['weight_decay']}" if pd.notna(row.get('weight_decay')) else 'N/A'
    dropout = str(row.get('dropout', 'N/A'))
    if dropout == 'nan':
        dropout = 'N/A'
    emb_dim = f"{int(row['embedding_dim'])}" if pd.notna(row.get('embedding_dim')) else 'N/A'
    
    # Truncate dropout if too long
    if len(dropout) > 25:
        dropout = '0.5/0.5/0.4'
    
    datetime_short = row['run_datetime'][:8] + '...'
    
    row_values = [
        f"{row['run_num']}", 
        datetime_short,
        test_acc,
        val_acc,
        best_val_acc,
        test_loss,
        epochs,
        completed,
        lr,
        batch,
        wd,
        dropout,
        emb_dim,
        ', '.join(key_changes)
    ]
    
    print('| ' + ' | '.join(row_values) + ' |')

# Print statistics
print("\n\n### Run Statistics\n")
complete_runs = df[df['completed'] == 'Yes']
print(f"- **Total Runs**: {len(df)}")
print(f"- **Completed Runs**: {len(complete_runs)}")
print(f"- **Incomplete Runs**: {len(df) - len(complete_runs)}")

if len(complete_runs) > 0:
    best_run = complete_runs.loc[complete_runs['test_acc'].idxmax()]
    print(f"- **Best Test Accuracy**: {best_run['test_acc']:.2f}% (Run #{best_run['run_num']})")
    print(f"- **Mean Test Accuracy** (completed runs): {complete_runs['test_acc'].mean():.2f}% ± {complete_runs['test_acc'].std():.2f}%")
    print(f"- **Best Validation Accuracy**: {complete_runs['best_val_acc'].max():.2f}%")

print("\n\n### Key Changes Between Runs\n")

# Analyze changes run by run
for idx in range(len(df) - 1):
    curr = df.iloc[idx]
    next_run = df.iloc[idx + 1]
    
    changes = []
    
    # Check what changed
    if curr.get('weight_decay') != next_run.get('weight_decay'):
        changes.append(f"Weight decay: {curr.get('weight_decay')} → {next_run.get('weight_decay')}")
    
    if curr.get('dropout') != next_run.get('dropout'):
        changes.append(f"Dropout changed")
    
    if curr.get('architecture') != next_run.get('architecture'):
        changes.append(f"Architecture updated")
    
    if curr.get('improvements') != next_run.get('improvements'):
        changes.append(f"Improvements: {next_run.get('improvements', 'N/A')}")
    
    if curr.get('embedding_dim') != next_run.get('embedding_dim'):
        changes.append(f"Embedding: {curr.get('embedding_dim')} → {next_run.get('embedding_dim')}")
    
    if changes:
        print(f"\n**Run {curr['run_num']} → Run {next_run['run_num']}:**")
        for change in changes:
            print(f"  - {change}")

print("\n")
