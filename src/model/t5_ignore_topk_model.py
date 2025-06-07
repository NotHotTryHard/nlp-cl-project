import torch
import numpy as np

from src.model.t5model import T5forSummarization

class T5IgnoreTopK(T5forSummarization):
    def __init__(
        self,
        model_name,
        cache_dir,
        max_length,
        tau=20,                      # steps before applying pruning
        prune_fraction_k=0.08,       # prune fraction 
        global_pruning=True,         # global or per-tensor pruning
        threshold_tol=1e-5,          # tolerance for binary search threshold accuracy
        max_tensor_size=10_000_000,  # max size for torch.quantile, use binary search if larger
        output_hidden_states=True,
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            max_length=max_length,
            output_hidden_states=output_hidden_states
        )
        
        self.tau = tau
        self.prune_fraction_k = prune_fraction_k
        self.global_pruning = global_pruning
        self.threshold_tol = threshold_tol
        self.max_tensor_size = max_tensor_size
        
        self.theta_base = None
        self.step_counter = 0
        self.has_snapshot = False
        self.training_initialized = False
        print(f"[TOPK] Initialized with tau={tau}, prune_fraction={prune_fraction_k}, global_pruning={global_pruning}, threshold_tol={threshold_tol}")
    
    def snapshot_base_weights(self):
        self.theta_base = {}
        device = next(self.parameters()).device
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.theta_base[name] = param.clone().detach().to(device)
        
        self.step_counter = 0
        self.has_snapshot = True
        # print(f"[TOPK] Snapshot taken: captured {len(self.theta_base)} parameter tensors")
    
    def find_kth_value(self, all_deltas, k_fraction, tensor_name=None):
        name_str = f" for tensor {tensor_name}" if tensor_name else ""
        #print(f"[TOPK] Finding the k-th value for {k_fraction*100:.2f}% threshold{name_str}")
        
        total_elements = sum(delta.numel() for delta in all_deltas)
        max_val = max(delta.max().item() for delta in all_deltas)
        min_val = min(delta.min().item() for delta in all_deltas)
        
        #print(f"[TOPK] Total elements: {total_elements}")
        #print(f"[TOPK] Value range: [{min_val:.6f}, {max_val:.6f}]")
        
        # Try to use torch.quantile for small tensors
        if total_elements <= self.max_tensor_size and len(all_deltas) == 1:
            try:
                threshold = torch.quantile(all_deltas[0], 1 - k_fraction).item()
                #print(f"[TOPK] Used torch.quantile, threshold: {threshold:.6f}")
                return threshold
            except RuntimeError as e:
                if "quantile() input tensor is too large" in str(e):
                    print(f"[TOPK] Quantile failed, falling back to binary search")
                else:
                    raise e
        
        k_idx = int(total_elements * k_fraction)
        tol = self.threshold_tol
        left, right = min_val, max_val
        
        #print(f"[TOPK] Finding the value at index {k_idx} of {total_elements} (tolerance: {tol})")
        
        # binary search to find threshold
        iterations = 0
        max_iterations = int(np.log2((max_val - min_val) / tol)) + 10
        
        while right - left > tol and iterations < max_iterations:
            mid = (left + right) / 2
            count = sum((delta >= mid).sum().item() for delta in all_deltas)
            
            if count >= k_idx:
                left = mid
            else:
                right = mid
                
            iterations += 1
        
        threshold = (left + right) / 2
        # print(f"[TOPK] Found threshold: {threshold:.6f} after {iterations} iterations")
        # print(f"[TOPK] Approximation error bound: +/- {(right-left)/2:.8f}")
        return threshold
        
    def apply_ignore_topk_pruning(self):
        if self.theta_base is None:
            print("[TOPK] Warning: No snapshot available")
            return
        
        print(f"[TOPK] Applying pruning after {self.step_counter} steps...")
        param_count = 0
        total_params = 0
        pruned_params = 0
        
        if self.global_pruning:
            # Gather all deltas without concatenating them
            all_deltas = []
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.theta_base:
                    delta = (param.data - self.theta_base[name]).abs()
                    all_deltas.append(delta.flatten())
                    total_params += delta.numel()
            
            # Find the global threshold using binary search
            global_thr = self.find_kth_value(all_deltas, self.prune_fraction_k)
            #print(f"[TOPK] Global threshold: {global_thr:.6f}")
            
            # Apply the global threshold to each parameter
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.theta_base:
                    delta = param.data - self.theta_base[name]
                    mask = (delta.abs() <= global_thr).float()
                    theta_pruned = self.theta_base[name] + delta * mask
                    param.data.copy_(theta_pruned)
                    
                    pruned_count = (mask == 0).sum().item()
                    pruned_params += pruned_count
                    param_count += 1
        else:
            # per-tensor pruning
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.theta_base:
                    delta = param.data - self.theta_base[name]
                    delta_abs = delta.abs()
                    total_params += delta.numel()
                    
                    thr = self.find_kth_value([delta_abs.flatten()], self.prune_fraction_k, name)
                    
                    mask = (delta_abs <= thr).float()
                    theta_pruned = self.theta_base[name] + delta * mask
                    param.data.copy_(theta_pruned)
                    
                    pruned_count = (mask == 0).sum().item()
                    pruned_params += pruned_count
                    param_count += 1
        
        print(f"[TOPK] Pruning complete: modified {param_count} tensors, pruned {pruned_params}/{total_params} ({100.0 * pruned_params / total_params:.2f}%) weights")
        
        self.step_counter = 0
        self.theta_base = None  
        self.has_snapshot = False
        #print("[TOPK] Taking new snapshot for next pruning cycle")
        self.snapshot_base_weights()
    
    def init_for_training(self):
        #print("[TOPK] Training mode activated")
        if not self.has_snapshot:
            #print("[TOPK] Taking new snapshot of weights")
            self.snapshot_base_weights()
        self.training_initialized = True
    
    def forward(self, batch):
        outputs = super().forward(batch)
        
        if self.training and not self.training_initialized:
            self.init_for_training()
        
        if self.training and self.has_snapshot:
            self.step_counter += 1
            #if self.step_counter % max(1, self.tau // 4) == 0:
                #print(f"[TOPK] Step counter: {self.step_counter}/{self.tau}")
            
            if self.step_counter >= self.tau:
                self.apply_ignore_topk_pruning()
        
        return outputs
