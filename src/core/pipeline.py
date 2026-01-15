"""
AegisVerity Forensic Pipeline
Main orchestration system for layered forensic analysis
"""

from typing import Dict, List, Any, Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_types import (
    ForensicResult, 
    DetectionConfig, 
    LayerOutput, 
    FusionResult, 
    MediaMetadata,
    DetectionStatus
)
from .base_layer import BaseDetectionLayer


class ForensicPipeline:
    """
    Main forensic analysis pipeline orchestrator
    
    Manages execution of multiple detection layers and result fusion
    """
    
    def __init__(self, layers: List[BaseDetectionLayer], config: DetectionConfig):
        self.layers = layers
        self.config = config
        self.pipeline_id = f"pipeline_{int(time.time())}"
        self.execution_history: List[Dict[str, Any]] = []
        
    def analyze_media(
        self, 
        media_path: str, 
        metadata: MediaMetadata,
        parallel_execution: bool = True
    ) -> FusionResult:
        """
        Run complete forensic analysis pipeline
        
        Args:
            media_path: Path to media file
            metadata: Media metadata
            parallel_execution: Whether to run layers in parallel
            
        Returns:
            FusionResult with combined analysis
        """
        start_time = time.time()
        
        # Load all models
        self._load_all_models()
        
        # Execute all layers
        layer_outputs = self._execute_layers(media_path, metadata, parallel_execution)
        
        # Fuse results
        fusion_result = self._fuse_results(layer_outputs)
        
        # Record execution
        execution_time = time.time() - start_time
        self._record_execution(media_path, layer_outputs, fusion_result, execution_time)
        
        return fusion_result
    
    def _load_all_models(self) -> None:
        """Load models for all layers"""
        print(f"🔄 Loading models for {len(self.layers)} detection layers...")
        
        for layer in self.layers:
            try:
                success = layer.load_models()
                status = "✅" if success else "❌"
                print(f"  {status} {layer.layer_name}")
            except Exception as e:
                print(f"  ❌ {layer.layer_name}: {str(e)}")
    
    def _execute_layers(
        self, 
        media_path: str, 
        metadata: MediaMetadata,
        parallel: bool
    ) -> Dict[str, LayerOutput]:
        """Execute all detection layers"""
        layer_outputs = {}
        
        if parallel and len(self.layers) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(len(self.layers), 4)) as executor:
                # Submit all layer tasks
                future_to_layer = {
                    executor.submit(layer.analyze, media_path, metadata): layer
                    for layer in self.layers
                }
                
                # Collect results
                for future in as_completed(future_to_layer):
                    layer = future_to_layer[future]
                    try:
                        output = future.result()
                        layer_outputs[layer.layer_id] = output
                        print(f"  ✅ {layer.layer_name}: {output.aggregated_confidence:.3f} confidence")
                    except Exception as e:
                        print(f"  ❌ {layer.layer_name}: {str(e)}")
                        # Create error output
                        error_output = self._create_error_output(layer, str(e))
                        layer_outputs[layer.layer_id] = error_output
        else:
            # Sequential execution
            for layer in self.layers:
                try:
                    output = layer.analyze(media_path, metadata)
                    layer_outputs[layer.layer_id] = output
                    print(f"  ✅ {layer.layer_name}: {output.aggregated_confidence:.3f} confidence")
                except Exception as e:
                    print(f"  ❌ {layer.layer_name}: {str(e)}")
                    error_output = self._create_error_output(layer, str(e))
                    layer_outputs[layer.layer_id] = error_output
        
        return layer_outputs
    
    def _fuse_results(self, layer_outputs: Dict[str, LayerOutput]) -> FusionResult:
        """
        Fuse results from multiple layers using weighted consensus
        
        Args:
            layer_outputs: Results from all layers
            
        Returns:
            FusionResult with combined analysis
        """
        if not layer_outputs:
            return FusionResult(
                final_status=DetectionStatus.ERROR,
                final_confidence=0.0,
                layer_outputs={},
                fusion_method="none",
                explanation="No layer outputs to fuse",
                supporting_evidence={},
                consensus_score=0.0
            )
        
        # Extract confidence scores from all layers
        layer_confidences = {
            layer_id: output.aggregated_confidence 
            for layer_id, output in layer_outputs.items()
        }
        
        # Calculate weighted consensus
        weights = self._get_layer_weights()
        weighted_confidence = sum(
            conf * weights.get(layer_id, 1.0) 
            for layer_id, conf in layer_confidences.items()
        ) / sum(weights.values())
        
        # Determine final status
        final_status = self._determine_final_status(weighted_confidence, layer_outputs)
        
        # Generate explanation
        explanation = self._generate_explanation(layer_outputs, weighted_confidence, final_status)
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(layer_outputs)
        
        # Collect supporting evidence
        supporting_evidence = self._collect_supporting_evidence(layer_outputs)
        
        return FusionResult(
            final_status=final_status,
            final_confidence=weighted_confidence,
            layer_outputs=layer_outputs,
            fusion_method="weighted_consensus",
            explanation=explanation,
            supporting_evidence=supporting_evidence,
            consensus_score=consensus_score
        )
    
    def _get_layer_weights(self) -> Dict[str, float]:
        """Get weights for each layer in fusion"""
        # Default weights - can be customized per deployment
        default_weights = {
            "l1_forensic": 0.15,    # Forensic analysis
            "l2_visual": 0.25,       # Visual analysis
            "l3_audio_visual": 0.20, # Audio-visual analysis
            "l4_audio": 0.20,        # Audio analysis
            "l5_explainability": 0.10, # Explainability
            "l6_fusion": 0.10          # Fusion layer
        }
        
        # Filter to only include active layers
        active_weights = {}
        for layer in self.layers:
            active_weights[layer.layer_id] = default_weights.get(
                layer.layer_id.replace("l", "l"), 1.0
            )
        
        return active_weights
    
    def _determine_final_status(
        self, 
        confidence: float, 
        layer_outputs: Dict[str, LayerOutput]
    ) -> DetectionStatus:
        """Determine final status based on confidence and layer consensus"""
        # Primary confidence-based determination
        if confidence >= 0.8:
            return DetectionStatus.MANIPULATED
        elif confidence >= 0.6:
            return DetectionStatus.SUSPICIOUS
        elif confidence >= 0.4:
            return DetectionStatus.AUTHENTIC
        else:
            return DetectionStatus.UNKNOWN
    
    def _generate_explanation(
        self, 
        layer_outputs: Dict[str, LayerOutput], 
        confidence: float,
        status: DetectionStatus
    ) -> str:
        """Generate human-readable explanation of results"""
        explanations = []
        
        # Add layer-specific explanations
        for layer_id, output in layer_outputs.items():
            if output.anomalies:
                explanations.append(f"{output.layer_name}: {', '.join(output.anomalies[:2])}")
        
        # Add confidence explanation
        explanations.append(f"Overall confidence: {confidence:.1%}")
        
        # Add status explanation
        status_explanations = {
            DetectionStatus.AUTHENTIC: "Content appears authentic with high confidence",
            DetectionStatus.SUSPICIOUS: "Content shows suspicious patterns requiring review",
            DetectionStatus.MANIPULATED: "Content likely manipulated or synthetic",
            DetectionStatus.ERROR: "Analysis encountered technical errors",
            DetectionStatus.UNKNOWN: "Unable to determine authenticity"
        }
        
        explanations.append(status_explanations.get(status, "Unknown status"))
        
        return " | ".join(explanations)
    
    def _calculate_consensus(self, layer_outputs: Dict[str, LayerOutput]) -> float:
        """Calculate consensus score among layers"""
        if len(layer_outputs) < 2:
            return 1.0
        
        # Get status from each layer
        layer_statuses = []
        for output in layer_outputs.values():
            if output.results:
                best_result = output.get_best_result()
                if best_result:
                    layer_statuses.append(best_result.status.value)
        
        if not layer_statuses:
            return 0.0
        
        # Calculate agreement
        status_counts = {}
        for status in layer_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Consensus is the proportion of layers agreeing with the majority
        majority_count = max(status_counts.values())
        consensus = majority_count / len(layer_statuses)
        
        return consensus
    
    def _collect_supporting_evidence(
        self, 
        layer_outputs: Dict[str, LayerOutput]
    ) -> Dict[str, Any]:
        """Collect supporting evidence from all layers"""
        evidence = {
            "layer_count": len(layer_outputs),
            "total_anomalies": 0,
            "processing_times": {},
            "confidence_distribution": {},
            "key_findings": []
        }
        
        for layer_id, output in layer_outputs.items():
            # Collect anomalies
            evidence["total_anomalies"] += len(output.anomalies)
            
            # Collect processing times
            evidence["processing_times"][layer_id] = output.processing_time
            
            # Collect confidence scores
            evidence["confidence_distribution"][layer_id] = output.aggregated_confidence
            
            # Collect key findings
            if output.anomalies:
                evidence["key_findings"].extend([
                    f"{output.layer_name}: {anomaly}" 
                    for anomaly in output.anomalies[:3]  # Top 3 per layer
                ])
        
        return evidence
    
    def _create_error_output(self, layer: BaseDetectionLayer, error_msg: str) -> LayerOutput:
        """Create error output for failed layer"""
        return LayerOutput(
            layer_id=layer.layer_id,
            layer_name=layer.layer_name,
            results=[],
            processing_time=0.0,
            confidence_scores=[],
            aggregated_confidence=0.0,
            anomalies=[f"Layer error: {error_msg}"]
        )
    
    def _record_execution(
        self, 
        media_path: str, 
        layer_outputs: Dict[str, LayerOutput],
        fusion_result: FusionResult,
        execution_time: float
    ) -> None:
        """Record pipeline execution for audit trail"""
        execution_record = {
            "pipeline_id": self.pipeline_id,
            "media_path": media_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": execution_time,
            "layer_count": len(self.layers),
            "layer_outputs": {
                layer_id: {
                    "confidence": output.aggregated_confidence,
                    "processing_time": output.processing_time,
                    "anomaly_count": len(output.anomalies)
                }
                for layer_id, output in layer_outputs.items()
            },
            "fusion_result": {
                "final_status": fusion_result.final_status.value,
                "final_confidence": fusion_result.final_confidence,
                "consensus_score": fusion_result.consensus_score,
                "fusion_method": fusion_result.fusion_method
            }
        }
        
        self.execution_history.append(execution_record)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get pipeline execution history"""
        return self.execution_history.copy()
    
    def cleanup(self) -> None:
        """Clean up all layers"""
        print("🧹 Cleaning up forensic pipeline...")
        
        for layer in self.layers:
            try:
                layer.cleanup()
                print(f"  ✅ {layer.layer_name}")
            except Exception as e:
                print(f"  ❌ {layer.layer_name}: {str(e)}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_id": self.pipeline_id,
            "layer_count": len(self.layers),
            "layers": [
                {
                    "id": layer.layer_id,
                    "name": layer.layer_name,
                    "config": layer.get_layer_info()
                }
                for layer in self.layers
            ],
            "execution_count": len(self.execution_history),
            "last_execution": self.execution_history[-1] if self.execution_history else None
        }
