export interface ProfileSample {
  timestamp_ms: number;
  token_id: number;
  wall_ms: number;
  io_wait_ms: number;
  matmul_ms: number;
  attention_ms: number;
  lm_head_ms: number;
  tok_per_sec: number;
  cache_hits: number;
  disk_reads: number;
  cache_routed: number;
}

export interface ExpertTelemetry {
  layer: number;
  expert: number;
  activations: number;
  lfru_score: number;
  tier: 'VRAM' | 'RAM' | 'DISK';
  pinned: boolean;
  last_prob: number;
  topic: string;
}

export interface EngineHealth {
  status: string;
  model_name: string;
  architecture: string;
  format: string;
  active_slots: number;
  queued_requests: number;
  hparams: {
    vocab_size: number;
    dim: number;
    n_layers: number;
    n_heads: number;
    n_kv_heads: number;
    n_experts: number;
    n_experts_used: number;
    max_seq_len: number;
  };
  hardware: {
    cpu_brand: string;
    logical_cores: number;
    simd_backend: string;
    io_backend: string;
    total_ram_bytes: number;
    avail_ram_bytes: number;
    vram_budget_bytes: number;
  };
  tiers: {
    vram_experts: number;
    ram_experts: number;
    disk_experts: number;
    total_experts: number;
    dense_backbone_bytes: number;
    sparse_disk_bytes: number;
    resident_cache_bytes: number;
    cache_hits: number;
    cache_misses: number;
    cache_routed: number;
    bytes_streamed: number;
    io_ops: number;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  reasoning_content?: string;
  telemetry?: {
    tok_per_sec: number;
    wall_ms: number;
    io_wait_ms: number;
    cache_hits: number;
    disk_reads: number;
  };
}

export interface SystemOneScore {
  option: string;
  logit: number;
  probability: number;
  token_id: number;
}

export interface SystemOneResult {
  wall_ms: number;
  io_wait_ms: number;
  matmul_ms: number;
  generated_tokens: number;
  recommended: string;
  options: SystemOneScore[];
}

export interface ImagePreview {
  step: number;
  total_steps: number;
  sigma: number;
  url: string;
}

export interface ImageResult {
  elapsed_ms: number;
  width: number;
  height: number;
  steps: number;
  guidance_scale: number;
  seed: number;
  data: { url: string; prompt: string }[];
  previews: ImagePreview[];
}

