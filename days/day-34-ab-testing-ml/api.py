#!/usr/bin/env python3
"""
Day 34: A/B Testing API - Production REST API

FastAPI-based REST API for managing A/B testing experiments, providing endpoints
for experiment management, user assignment, result recording, and real-time analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import uvicorn
from datetime import datetime
import logging

# Import A/B testing framework
from solution import (
    ExperimentConfig, ExperimentManager, ExperimentMonitor,
    StreamingPlatform, StatisticalAnalyzer
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="A/B Testing API",
    description="Production API for managing ML model A/B tests",
    version="1.0.0"
)

# Global instances
platform = StreamingPlatform()
analyzer = StatisticalAnalyzer()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ExperimentCreateRequest(BaseModel):
    """Request to create new experiment"""
    experiment_id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    variants: Dict[str, Dict[str, Any]] = Field(..., description="Experiment variants")
    traffic_allocation: Dict[str, float] = Field(..., description="Traffic allocation")
    primary_metric: str = Field(..., description="Primary success metric")
    secondary_metrics: List[str] = Field(default=[], description="Secondary metrics")
    significance_level: float = Field(0.05, description="Statistical significance level")
    power: float = Field(0.8, description="Statistical power")
    minimum_detectable_effect: float = Field(0.05, description="Minimum detectable effect")
    min_sample_size: int = Field(1000, description="Minimum sample size")
    guardrail_metrics: Dict[str, Dict[str, float]] = Field(default={}, description="Guardrail metrics")
    eligibility_criteria: Dict[str, Any] = Field(default={}, description="User eligibility criteria")

class UserAssignmentRequest(BaseModel):
    """Request for user assignment"""
    experiment_id: str = Field(..., description="Experiment ID")
    user_id: str = Field(..., description="User ID")
    user_attributes: Dict[str, Any] = Field(default={}, description="User attributes")

class ResultRecordRequest(BaseModel):
    """Request to record experiment result"""
    experiment_id: str = Field(..., description="Experiment ID")
    user_id: str = Field(..., description="User ID")
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")

class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: str = Field(..., description="User ID")
    user_attributes: Dict[str, Any] = Field(default={}, description="User attributes")
    experiment_id: Optional[str] = Field(None, description="Experiment ID for A/B testing")
    n_recommendations: int = Field(10, description="Number of recommendations")

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "A/B Testing API for ML Models",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_experiments": len(platform.experiment_manager.active_experiments)
    }

# =============================================================================
# EXPERIMENT MANAGEMENT
# =============================================================================

@app.post("/experiments")
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new A/B test experiment"""
    try:
        # Create experiment config
        config = ExperimentConfig(
            experiment_id=request.experiment_id,
            name=request.name,
            description=request.description,
            variants=request.variants,
            traffic_allocation=request.traffic_allocation,
            primary_metric=request.primary_metric,
            secondary_metrics=request.secondary_metrics,
            significance_level=request.significance_level,
            power=request.power,
            minimum_detectable_effect=request.minimum_detectable_effect,
            min_sample_size=request.min_sample_size,
            guardrail_metrics=request.guardrail_metrics,
            eligibility_criteria=request.eligibility_criteria
        )
        
        # Create experiment
        experiment_id = platform.experiment_manager.create_experiment(config)
        
        logger.info(f"Created experiment: {experiment_id}")
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": "Experiment created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment"""
    try:
        success = platform.experiment_manager.start_experiment(experiment_id)
        
        if success:
            logger.info(f"Started experiment: {experiment_id}")
            return {
                "experiment_id": experiment_id,
                "status": "running",
                "message": "Experiment started successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to start experiment")
            
    except Exception as e:
        logger.error(f"Error starting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/experiments")
async def list_experiments():
    """List all experiments"""
    experiments = []
    
    for exp_id, exp_data in platform.experiment_manager.experiments.items():
        experiments.append({
            "experiment_id": exp_id,
            "name": exp_data['config'].name,
            "status": exp_data['status'].value,
            "created_at": exp_data['created_at'].isoformat()
        })
    
    return {"experiments": experiments}

@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    if experiment_id not in platform.experiment_manager.experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp_data = platform.experiment_manager.experiments[experiment_id]
    config = exp_data['config']
    
    return {
        "experiment_id": experiment_id,
        "name": config.name,
        "description": config.description,
        "status": exp_data['status'].value,
        "variants": config.variants,
        "traffic_allocation": config.traffic_allocation,
        "primary_metric": config.primary_metric,
        "created_at": exp_data['created_at'].isoformat()
    }

@app.get("/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str):
    """Get detailed experiment status and analysis"""
    try:
        status = platform.monitor.get_experiment_status(experiment_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting experiment status {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# USER ASSIGNMENT AND RESULTS
# =============================================================================

@app.post("/experiments/assign")
async def assign_user(request: UserAssignmentRequest):
    """Assign user to experiment variant"""
    try:
        variant = platform.experiment_manager.get_assignment(
            request.experiment_id,
            request.user_id,
            request.user_attributes
        )
        
        if variant:
            return {
                "experiment_id": request.experiment_id,
                "user_id": request.user_id,
                "variant": variant,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "experiment_id": request.experiment_id,
                "user_id": request.user_id,
                "variant": None,
                "message": "User not eligible for experiment"
            }
            
    except Exception as e:
        logger.error(f"Error assigning user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/experiments/results")
async def record_result(request: ResultRecordRequest):
    """Record experiment result"""
    try:
        success = platform.experiment_manager.record_result(
            request.experiment_id,
            request.user_id,
            request.metric_name,
            request.metric_value
        )
        
        if success:
            return {
                "status": "recorded",
                "experiment_id": request.experiment_id,
                "user_id": request.user_id,
                "metric_name": request.metric_name,
                "metric_value": request.metric_value,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to record result")
            
    except Exception as e:
        logger.error(f"Error recording result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get all results for an experiment"""
    try:
        results = platform.experiment_manager.get_experiment_results(experiment_id)
        
        # Convert datetime objects to ISO strings
        formatted_results = []
        for result in results:
            formatted_result = result.copy()
            formatted_result['timestamp'] = result['timestamp'].isoformat()
            formatted_results.append(formatted_result)
        
        return {
            "experiment_id": experiment_id,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error getting results for {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RECOMMENDATIONS WITH A/B TESTING
# =============================================================================

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations with A/B testing"""
    try:
        result = platform.get_recommendations_for_user(
            request.user_id,
            request.user_attributes,
            request.experiment_id
        )
        
        return {
            "user_id": request.user_id,
            "recommendations": result['recommendations'][:request.n_recommendations],
            "model_used": result['model_used'],
            "experiment_id": result.get('experiment_id'),
            "variant": result.get('variant'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

@app.post("/analysis/continuous")
async def analyze_continuous_metric(
    control_data: List[float],
    treatment_data: List[float],
    significance_level: float = 0.05
):
    """Analyze continuous metric data"""
    try:
        import numpy as np
        
        analyzer = StatisticalAnalyzer(significance_level)
        result = analyzer.analyze_continuous_metric(
            np.array(control_data),
            np.array(treatment_data)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in continuous analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/binary")
async def analyze_binary_metric(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    significance_level: float = 0.05
):
    """Analyze binary metric data"""
    try:
        analyzer = StatisticalAnalyzer(significance_level)
        result = analyzer.analyze_binary_metric(
            control_successes, control_total,
            treatment_successes, treatment_total
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in binary analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analysis/sample-size")
async def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    power: float = 0.8,
    significance_level: float = 0.05
):
    """Calculate required sample size"""
    try:
        analyzer = StatisticalAnalyzer(significance_level)
        sample_size = analyzer.calculate_sample_size(
            baseline_rate, minimum_detectable_effect, power, significance_level
        )
        
        return {
            "baseline_rate": baseline_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "power": power,
            "significance_level": significance_level,
            "required_sample_size_per_variant": sample_size,
            "total_sample_size": sample_size * 2
        }
        
    except Exception as e:
        logger.error(f"Error calculating sample size: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================

@app.post("/experiments/{experiment_id}/simulate")
async def simulate_experiment(
    experiment_id: str,
    n_users: int = 1000,
    n_days: int = 7,
    background_tasks: BackgroundTasks = None
):
    """Run experiment simulation"""
    try:
        if experiment_id not in platform.experiment_manager.experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        config = platform.experiment_manager.experiments[experiment_id]['config']
        
        # Run simulation
        results = platform.run_experiment_simulation(config, n_users, n_days)
        
        return {
            "experiment_id": experiment_id,
            "simulation_parameters": {
                "n_users": n_users,
                "n_days": n_days
            },
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error running simulation for {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# METRICS AND MONITORING
# =============================================================================

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    return {
        "total_experiments": len(platform.experiment_manager.experiments),
        "active_experiments": len(platform.experiment_manager.active_experiments),
        "total_results": len(platform.experiment_manager.results_log),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get metrics for specific experiment"""
    try:
        if experiment_id not in platform.experiment_manager.active_experiments:
            raise HTTPException(status_code=404, detail="Active experiment not found")
        
        experiment = platform.experiment_manager.active_experiments[experiment_id]
        results = experiment['results']
        
        # Calculate basic metrics
        total_users = len(set(r['user_id'] for r in results))
        total_events = len(results)
        
        # Group by variant
        variant_stats = {}
        for result in results:
            variant = result['variant']
            if variant not in variant_stats:
                variant_stats[variant] = {'users': set(), 'events': 0}
            
            variant_stats[variant]['users'].add(result['user_id'])
            variant_stats[variant]['events'] += 1
        
        # Convert sets to counts
        for variant in variant_stats:
            variant_stats[variant]['users'] = len(variant_stats[variant]['users'])
        
        return {
            "experiment_id": experiment_id,
            "total_users": total_users,
            "total_events": total_events,
            "variant_statistics": variant_stats,
            "runtime_hours": (datetime.now() - experiment['start_time']).total_seconds() / 3600
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics for {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the API server"""
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()