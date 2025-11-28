import os
from sklearn.preprocessing import LabelEncoder
from src.datasets import find_file_pairs
import tqdm
import pandas as pd
def splitsave_dataframes(files_list):
    fitted = False
    fwd_encoder = LabelEncoder()
    bwd_encoder = LabelEncoder()
    for nodes_path, edges_path in tqdm.tqdm(files_list):
        nodes_df = pd.read_csv(nodes_path, sep='\t')
        edges_df = pd.read_csv(edges_path, sep='\t')
        
        # Разделение датасетов на fwd и bwd
        edges_df['Vid_fwd'] = edges_df['Vid'] == 0
        edges_df['Vid_bwd'] = edges_df['Vid'] == 1
        edges_df['Vid_usr'] = edges_df['Vid'] == 2
        fwd_edges = edges_df.loc[edges_df['Vid_fwd'] | edges_df['Vid_usr']].copy()
        bwd_edges = edges_df.loc[edges_df['Vid_bwd'] | edges_df['Vid_usr']].copy()
        fwd_ids = set(pd.concat([fwd_edges['id_in'], fwd_edges['id_out']]))
        bwd_ids = set(pd.concat([bwd_edges['id_in'], bwd_edges['id_out']]))
        
        #fixing 5 -> 186 -> 2 into 5 -> 2 edges
        bwd_edges.loc[bwd_edges['id_out'] == 186, 'l'] += bwd_edges.loc[bwd_edges['id_in'] == 186, 'l'].values[0]
        bwd_edges.loc[bwd_edges['id_out'] == 186, 'id_out'] = bwd_edges.loc[bwd_edges['id_in'] == 186, 'id_out'].values[0]
        bwd_edges = bwd_edges.drop(bwd_edges.loc[bwd_edges['id_in'] == 186].index)
        
        fwd_nodes = nodes_df[nodes_df['id'].isin(fwd_ids)].copy()
        bwd_nodes = nodes_df[nodes_df['id'].isin(bwd_ids)].copy()
        
        # replacing 5 node with 186 and deleting latter because 186 is imaginary node
        bwd_nodes.loc[bwd_nodes['id'] == 5, ['Q', 'P', 'Temp']] = bwd_nodes.loc[bwd_nodes['id'] == 186, ['Q', 'P', 'Temp']].values
        bwd_nodes = bwd_nodes.drop(bwd_nodes[bwd_nodes['id'] == 186].index)
        
        if not fitted:
            fwd_encoder.fit(fwd_nodes['id'])
            bwd_encoder.fit(bwd_nodes['id'])
            fitted = True
        fwd_path = edges_path.replace(".csv", "_fwd.csv")
        bwd_path = edges_path.replace(".csv", "_bwd.csv")
        
        fwd_path = fwd_path.replace("Termo_model", "Termo_model_fwd")
        bwd_path = bwd_path.replace("Termo_model", "Termo_model_bwd")
        
        fwd_edges = fwd_edges.drop(columns=['Vid_fwd', 'Vid_bwd', 'Vid_usr'])
        bwd_edges = bwd_edges.drop(columns=['Vid_fwd', 'Vid_bwd', 'Vid_usr'])
        
        fwd_nodes['id'] = fwd_encoder.transform(fwd_nodes['id'])
        bwd_nodes['id'] = bwd_encoder.transform(bwd_nodes['id'])
                
        fwd_edges['id_in'] = fwd_encoder.transform(fwd_edges['id_in'])
        fwd_edges['id_out'] = fwd_encoder.transform(fwd_edges['id_out'])
        bwd_edges['id_in'] = bwd_encoder.transform(bwd_edges['id_in'])
        bwd_edges['id_out'] = bwd_encoder.transform(bwd_edges['id_out'])
        bwd_edges.loc[bwd_edges['Vid'] == 2, ['id_in', 'id_out']]  = bwd_edges.loc[bwd_edges['Vid'] == 2, ['id_out', 'id_in']].values
        bwd_edges[['id_in', 'id_out']] = bwd_edges[['id_out', 'id_in']]  
        os.makedirs(os.path.dirname(fwd_path), exist_ok=True)
        os.makedirs(os.path.dirname(bwd_path), exist_ok=True)
        fwd_edges.to_csv(fwd_path, sep='\t')
        bwd_edges.to_csv(bwd_path, sep='\t')
        fwd_path = fwd_path.replace("tubes", "nodes")
        bwd_path = bwd_path.replace("tubes", "nodes")
        fwd_nodes.to_csv(fwd_path, index=False, sep='\t')
        bwd_nodes.to_csv(bwd_path, index=False, sep='\t')
        
        # Отдельное сохранение глобальных параметров
        global_nodes = pd.concat([fwd_nodes[fwd_nodes['types'] == 1], bwd_nodes[bwd_nodes['types'] == 2]])
        global_path = nodes_path.replace("nodes", "global")
        for direction in 'fwd', 'bwd':
            global_path_dir = global_path.replace("Termo_model",f"Termo_model_{direction}").replace(".csv", f"_{direction}.csv")
            global_nodes_dir = global_nodes.copy()
            global_nodes_dir['channel'] = 0 if direction == 'fwd' else 1
            global_nodes_dir.to_csv(global_path_dir, index= False, sep= '\t')
            
if __name__ == "__main__":
    print("Разделение IDEAL данных...")
    splitsave_dataframes(find_file_pairs('./datasets/Termo_model', True))
    print("Разделение зашумленных данных...")
    splitsave_dataframes(find_file_pairs('./datasets/Termo_model', False))