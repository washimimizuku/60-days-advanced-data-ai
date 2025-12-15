"""
Day 9: Data Lineage Tracking - Complete Solution
Production-ready implementation of comprehensive data lineage tracking system

This solution provides complete implementations for all exercises:
1. SQL parsing for automated lineage extraction
2. Multi-hop lineage tracking system
3. Column-level lineage with transformations
4. Impact analysis for change management
5. GDPR compliance lineage tracker
6. Data quality root cause analyzer
"""

import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    id: str
    type: str  # 'table', 'column', 'transformation'
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph"""
    source: str
    target: str
    transformation_type: str
    transformation_logic: str
    created_at: datetime

class SQLLineageExtractor:
    """
    Solution 1: Complete SQL parsing for automated lineage extraction
    """
    
    def __init__(self):
        self.table_patterns = {
            'select_from': r'FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?)',
            'join': r'(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?)',
            'insert_into': r'INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'update': r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'delete_from': r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'with_cte': r'WITH\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS'
        }
        
        self.column_patterns = {
            'select_columns': r'SELECT\s+(.*?)\s+FROM',
            'function_calls': r'([A-Z_]+)\s*\(\s*([^)]*)\s*\)',
            'column_alias': r'([a-zA-Z_][a-zA-Z0-9_.]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)'
        }
    
    def extract_table_lineage(self, sql_query: str) -> Dict[str, List[str]]:
        """Complete table-level lineage extraction from SQL query"""
        sql_clean = re.sub(r'--.*?\n', ' ', sql_query)  # Remove comments
        sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)  # Remove block comments
        sql_clean = re.sub(r'\s+', ' ', sql_clean).strip().upper()  # Normalize whitespace
        
        lineage = {
            'sources': [],
            'targets': []
        }
        
        # Extract source tables
        for pattern_name, pattern in self.table_patterns.items():
            if pattern_name in ['select_from', 'join']:
                matches = re.findall(pattern, sql_clean, re.IGNORECASE)
                for match in matches:
                    table_name = match.split()[0] if ' ' in match else match
                    table_name = table_name.lower()
                    if table_name not in lineage['sources']:
                        lineage['sources'].append(table_name)
        
        # Extract target tables
        for pattern_name, pattern in self.table_patterns.items():
            if pattern_name in ['insert_into', 'update', 'delete_from']:
                matches = re.findall(pattern, sql_clean, re.IGNORECASE)
                for match in matches:
                    table_name = match.split()[0] if ' ' in match else match
                    table_name = table_name.lower()
                    if table_name not in lineage['targets']:
                        lineage['targets'].append(table_name)
        
        # Handle CTEs as intermediate tables
        cte_matches = re.findall(self.table_patterns['with_cte'], sql_clean, re.IGNORECASE)
        for cte_name in cte_matches:
            lineage['sources'].append(cte_name.lower())
        
        return lineage
    
    def extract_column_lineage(self, sql_query: str) -> List[Dict[str, str]]:
        """Complete column-level lineage extraction from SQL query"""
        sql_clean = re.sub(r'--.*?\n', ' ', sql_query)
        sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)
        sql_clean = re.sub(r'\s+', ' ', sql_clean).strip()
        
        column_mappings = []
        
        # Extract SELECT clause
        select_match = re.search(self.column_patterns['select_columns'], sql_clean, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return column_mappings
        
        select_clause = select_match.group(1)
        
        # Split by comma, handling nested functions
        columns = self._split_select_columns(select_clause)
        
        for column_expr in columns:
            column_expr = column_expr.strip()
            
            # Check for function calls
            func_match = re.search(self.column_patterns['function_calls'], column_expr)
            if func_match:
                func_name = func_match.group(1)
                func_args = func_match.group(2)
                
                # Extract alias if present
                alias_match = re.search(r'\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*$', column_expr, re.IGNORECASE)
                target_column = alias_match.group(1) if alias_match else func_name.lower()
                
                # Parse function arguments for source columns
                if func_args.strip() == '*':
                    source_column = '*'
                    source_table = 'unknown'
                else:
                    source_parts = func_args.split('.')
                    if len(source_parts) == 2:
                        source_table, source_column = source_parts
                    else:
                        source_table = 'unknown'
                        source_column = func_args
                
                column_mappings.append({
                    'source_table': source_table.strip(),
                    'source_column': source_column.strip(),
                    'target_column': target_column,
                    'transformation': func_name
                })
            else:
                # Direct column mapping or alias
                alias_match = re.search(self.column_patterns['column_alias'], column_expr)
                if alias_match:
                    source_col = alias_match.group(1)
                    target_col = alias_match.group(2)
                    
                    source_parts = source_col.split('.')
                    if len(source_parts) == 2:
                        source_table, source_column = source_parts
                    else:
                        source_table = 'unknown'
                        source_column = source_parts[0]
                    
                    column_mappings.append({
                        'source_table': source_table,
                        'source_column': source_column,
                        'target_column': target_col,
                        'transformation': 'direct'
                    })
                else:
                    # Simple column reference
                    source_parts = column_expr.split('.')
                    if len(source_parts) == 2:
                        source_table, source_column = source_parts
                    else:
                        source_table = 'unknown'
                        source_column = source_parts[0]
                    
                    column_mappings.append({
                        'source_table': source_table,
                        'source_column': source_column,
                        'target_column': source_column,
                        'transformation': 'direct'
                    })
        
        return column_mappings
    
    def _split_select_columns(self, select_clause: str) -> List[str]:
        """Split SELECT clause by commas, handling nested functions and parentheses"""
        columns = []
        current_column = ""
        paren_depth = 0
        
        for char in select_clause:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                columns.append(current_column.strip())
                current_column = ""
                continue
            
            current_column += char
        
        if current_column.strip():
            columns.append(current_column.strip())
        
        return columns
    
    def parse_complex_query(self, sql_query: str) -> Dict[str, Any]:
        """Parse complex SQL with CTEs, subqueries, and window functions"""
        sql_clean = re.sub(r'--.*?\n', ' ', sql_query)
        sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)
        
        result = {
            'table_lineage': self.extract_table_lineage(sql_query),
            'column_lineage': self.extract_column_lineage(sql_query),
            'intermediate_steps': [],
            'complexity_score': 0
        }
        
        # Analyze complexity
        complexity_indicators = {
            'cte_count': len(re.findall(r'WITH\s+', sql_clean, re.IGNORECASE)),
            'subquery_count': len(re.findall(r'SELECT.*?FROM.*?\(.*?SELECT', sql_clean, re.IGNORECASE)),
            'join_count': len(re.findall(r'JOIN', sql_clean, re.IGNORECASE)),
            'window_function_count': len(re.findall(r'OVER\s*\(', sql_clean, re.IGNORECASE)),
            'aggregate_count': len(re.findall(r'(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(', sql_clean, re.IGNORECASE))
        }
        
        result['complexity_score'] = sum(complexity_indicators.values())
        result['intermediate_steps'] = list(complexity_indicators.keys())
        
        return result

class MultiHopLineageTracker:
    """
    Solution 2: Complete multi-hop lineage tracking system
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> LineageNode
        self.edges = []  # List of LineageEdge
        self.adjacency_list = defaultdict(list)  # source -> [targets]
        self.reverse_adjacency_list = defaultdict(list)  # target -> [sources]
    
    def add_node(self, node_id: str, node_type: str, metadata: Dict[str, Any]) -> None:
        """Add a node to the lineage graph"""
        node = LineageNode(
            id=node_id,
            type=node_type,
            metadata=metadata,
            created_at=datetime.now()
        )
        self.nodes[node_id] = node
        logger.info(f"Added node: {node_id} ({node_type})")
    
    def add_edge(self, source: str, target: str, transformation_type: str, 
                 transformation_logic: str) -> None:
        """Add an edge to the lineage graph"""
        edge = LineageEdge(
            source=source,
            target=target,
            transformation_type=transformation_type,
            transformation_logic=transformation_logic,
            created_at=datetime.now()
        )
        
        self.edges.append(edge)
        self.adjacency_list[source].append(target)
        self.reverse_adjacency_list[target].append(source)
        
        logger.info(f"Added edge: {source} -> {target} ({transformation_type})")
    
    def get_upstream_lineage(self, node_id: str, max_depth: int = 10) -> Set[str]:
        """Get all upstream dependencies for a node using BFS"""
        if node_id not in self.nodes:
            return set()
        
        upstream = set()
        queue = deque([(node_id, 0)])
        visited = {node_id}
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get all sources that feed into current node
            for source in self.reverse_adjacency_list[current_node]:
                if source not in visited:
                    upstream.add(source)
                    visited.add(source)
                    queue.append((source, depth + 1))
        
        return upstream
    
    def get_downstream_lineage(self, node_id: str, max_depth: int = 10) -> Set[str]:
        """Get all downstream dependencies for a node using BFS"""
        if node_id not in self.nodes:
            return set()
        
        downstream = set()
        queue = deque([(node_id, 0)])
        visited = {node_id}
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get all targets that current node feeds into
            for target in self.adjacency_list[current_node]:
                if target not in visited:
                    downstream.add(target)
                    visited.add(target)
                    queue.append((target, depth + 1))
        
        return downstream
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between two nodes using BFS"""
        if source not in self.nodes or target not in self.nodes:
            return []
        
        if source == target:
            return [source]
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current_node, path = queue.popleft()
            
            # Check all neighbors
            for neighbor in self.adjacency_list[current_node]:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def get_lineage_subgraph(self, node_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get a subgraph around a specific node"""
        upstream = self.get_upstream_lineage(node_id, depth)
        downstream = self.get_downstream_lineage(node_id, depth)
        
        all_nodes = upstream | downstream | {node_id}
        
        subgraph_nodes = {nid: asdict(self.nodes[nid]) for nid in all_nodes if nid in self.nodes}
        subgraph_edges = [
            asdict(edge) for edge in self.edges 
            if edge.source in all_nodes and edge.target in all_nodes
        ]
        
        return {
            'center_node': node_id,
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'upstream_count': len(upstream),
            'downstream_count': len(downstream)
        }

class ColumnLevelLineageTracker:
    """
    Solution 3: Complete column-level lineage with transformations
    """
    
    def __init__(self):
        self.column_mappings = []
        self.transformation_catalog = {}
        self.pii_patterns = ['email', 'phone', 'ssn', 'name', 'address', 'dob']
    
    def add_column_mapping(self, source_table: str, source_column: str,
                          target_table: str, target_column: str,
                          transformation_type: str, transformation_logic: str) -> None:
        """Add column-level lineage mapping"""
        mapping = {
            'source_table': source_table,
            'source_column': source_column,
            'target_table': target_table,
            'target_column': target_column,
            'transformation_type': transformation_type,
            'transformation_logic': transformation_logic,
            'created_at': datetime.now().isoformat()
        }
        
        self.column_mappings.append(mapping)
        
        # Add to transformation catalog for reuse
        catalog_key = f"{transformation_type}:{transformation_logic}"
        if catalog_key not in self.transformation_catalog:
            self.transformation_catalog[catalog_key] = {
                'type': transformation_type,
                'logic': transformation_logic,
                'usage_count': 0,
                'examples': []
            }
        
        self.transformation_catalog[catalog_key]['usage_count'] += 1
        self.transformation_catalog[catalog_key]['examples'].append({
            'source': f"{source_table}.{source_column}",
            'target': f"{target_table}.{target_column}"
        })
        
        logger.info(f"Added column mapping: {source_table}.{source_column} -> {target_table}.{target_column}")
    
    def trace_column_lineage(self, target_table: str, target_column: str) -> List[Dict[str, Any]]:
        """Trace lineage for a specific column back to its sources"""
        lineage_chain = []
        visited = set()
        
        def trace_recursive(table: str, column: str, depth: int = 0):
            if depth > 10 or f"{table}.{column}" in visited:
                return
            
            visited.add(f"{table}.{column}")
            
            # Find mappings where this column is the target
            for mapping in self.column_mappings:
                if (mapping['target_table'] == table and 
                    mapping['target_column'] == column):
                    
                    lineage_chain.append({
                        'step': len(lineage_chain) + 1,
                        'source_table': mapping['source_table'],
                        'source_column': mapping['source_column'],
                        'target_table': mapping['target_table'],
                        'target_column': mapping['target_column'],
                        'transformation_type': mapping['transformation_type'],
                        'transformation_logic': mapping['transformation_logic'],
                        'depth': depth
                    })
                    
                    # Recursively trace the source
                    trace_recursive(
                        mapping['source_table'],
                        mapping['source_column'],
                        depth + 1
                    )
        
        trace_recursive(target_table, target_column)
        return lineage_chain
    
    def find_pii_lineage(self) -> List[Dict[str, Any]]:
        """Find all columns that derive from PII data"""
        pii_lineage = []
        
        for mapping in self.column_mappings:
            source_col = mapping['source_column'].lower()
            target_col = mapping['target_column'].lower()
            
            # Check if source or target contains PII patterns
            is_pii_source = any(pattern in source_col for pattern in self.pii_patterns)
            is_pii_target = any(pattern in target_col for pattern in self.pii_patterns)
            
            if is_pii_source or is_pii_target:
                # Get full lineage chain for this PII column
                lineage_chain = self.trace_column_lineage(
                    mapping['target_table'],
                    mapping['target_column']
                )
                
                pii_info = {
                    'pii_type': self._identify_pii_type(source_col, target_col),
                    'source_table': mapping['source_table'],
                    'source_column': mapping['source_column'],
                    'target_table': mapping['target_table'],
                    'target_column': mapping['target_column'],
                    'transformation_type': mapping['transformation_type'],
                    'lineage_chain': lineage_chain,
                    'risk_level': self._assess_pii_risk(mapping),
                    'compliance_notes': self._get_compliance_notes(mapping)
                }
                
                pii_lineage.append(pii_info)
        
        return pii_lineage
    
    def _identify_pii_type(self, source_col: str, target_col: str) -> str:
        """Identify the type of PII data"""
        combined = f"{source_col} {target_col}".lower()
        
        for pattern in self.pii_patterns:
            if pattern in combined:
                return pattern
        return 'unknown_pii'
    
    def _assess_pii_risk(self, mapping: Dict[str, Any]) -> str:
        """Assess risk level of PII transformation"""
        transformation_type = mapping['transformation_type']
        
        high_risk_transformations = ['direct', 'concatenation', 'substring']
        medium_risk_transformations = ['hash', 'encryption', 'masking']
        low_risk_transformations = ['aggregation', 'count', 'exists']
        
        if transformation_type in high_risk_transformations:
            return 'high'
        elif transformation_type in medium_risk_transformations:
            return 'medium'
        elif transformation_type in low_risk_transformations:
            return 'low'
        else:
            return 'unknown'
    
    def _get_compliance_notes(self, mapping: Dict[str, Any]) -> List[str]:
        """Get compliance notes for PII transformation"""
        notes = []
        transformation_type = mapping['transformation_type']
        
        if transformation_type == 'direct':
            notes.append("Direct PII copy - ensure proper access controls")
            notes.append("Consider data retention policies")
        elif transformation_type == 'hash':
            notes.append("Hashed PII - verify hash algorithm strength")
        elif transformation_type == 'masking':
            notes.append("Masked PII - ensure masking is irreversible")
        elif transformation_type == 'aggregation':
            notes.append("Aggregated PII - lower privacy risk")
        
        return notes

class ImpactAnalyzer:
    """
    Solution 4: Complete impact analysis for change management
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.impact_cache = {}
        self.severity_weights = {
            'critical': 10,
            'high': 7,
            'medium': 4,
            'low': 1
        }
    
    def analyze_change_impact(self, changed_node: str, change_type: str) -> Dict[str, Any]:
        """Analyze impact of changes to a data asset"""
        cache_key = f"{changed_node}:{change_type}"
        if cache_key in self.impact_cache:
            return self.impact_cache[cache_key]
        
        # Get all downstream dependencies
        downstream_nodes = self.lineage.get_downstream_lineage(changed_node)
        
        impact_analysis = {
            'changed_node': changed_node,
            'change_type': change_type,
            'downstream_impacts': [],
            'severity_breakdown': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'affected_systems': [],
            'recommendations': [],
            'estimated_effort_hours': 0
        }
        
        # Analyze each downstream node
        for node_id in downstream_nodes:
            if node_id in self.lineage.nodes:
                node = self.lineage.nodes[node_id]
                severity = self._assess_impact_severity(changed_node, node_id, change_type)
                
                impact_info = {
                    'node_id': node_id,
                    'node_type': node.type,
                    'severity': severity,
                    'impact_description': self._generate_impact_description(change_type, node),
                    'mitigation_steps': self._generate_mitigation_steps(change_type, node),
                    'estimated_hours': self._estimate_effort_hours(severity, change_type)
                }
                
                impact_analysis['downstream_impacts'].append(impact_info)
                impact_analysis['severity_breakdown'][severity] += 1
                impact_analysis['affected_systems'].append(node_id)
                impact_analysis['estimated_effort_hours'] += impact_info['estimated_hours']
        
        # Generate recommendations
        impact_analysis['recommendations'] = self._generate_recommendations(
            change_type, impact_analysis['severity_breakdown'], downstream_nodes
        )
        
        # Cache the result
        self.impact_cache[cache_key] = impact_analysis
        
        return impact_analysis
    
    def _assess_impact_severity(self, source_node: str, target_node: str, change_type: str) -> str:
        """Assess severity of impact on a specific node"""
        if target_node not in self.lineage.nodes:
            return 'low'
        
        target = self.lineage.nodes[target_node]
        
        # Critical systems (production dashboards, ML models)
        if target.metadata.get('environment') == 'production':
            if change_type in ['removal', 'schema']:
                return 'critical'
            elif change_type in ['logic', 'data']:
                return 'high'
        
        # High impact (staging, analytics)
        elif target.metadata.get('environment') in ['staging', 'analytics']:
            if change_type in ['removal', 'schema']:
                return 'high'
            else:
                return 'medium'
        
        # Medium/Low impact (development, testing)
        else:
            if change_type == 'removal':
                return 'medium'
            else:
                return 'low'
    
    def _generate_impact_description(self, change_type: str, node: LineageNode) -> str:
        """Generate human-readable impact description"""
        descriptions = {
            'schema': f"Schema change may break queries/transformations in {node.id}",
            'data': f"Data changes may affect calculations and reports in {node.id}",
            'logic': f"Logic changes may alter business rules and outputs in {node.id}",
            'removal': f"Removal will break all dependencies on {node.id}"
        }
        return descriptions.get(change_type, f"Unknown change type impact on {node.id}")
    
    def _generate_mitigation_steps(self, change_type: str, node: LineageNode) -> List[str]:
        """Generate mitigation steps for the impact"""
        steps = {
            'schema': [
                "Update downstream queries to match new schema",
                "Test all transformations with new schema",
                "Update documentation and data contracts"
            ],
            'data': [
                "Validate data quality after changes",
                "Update data validation rules",
                "Notify downstream consumers of data changes"
            ],
            'logic': [
                "Review and update business logic",
                "Test all affected calculations",
                "Update unit tests and integration tests"
            ],
            'removal': [
                "Find alternative data sources",
                "Update all dependent systems",
                "Create migration plan for affected processes"
            ]
        }
        return steps.get(change_type, ["Review impact and create custom mitigation plan"])
    
    def _estimate_effort_hours(self, severity: str, change_type: str) -> int:
        """Estimate effort hours based on severity and change type"""
        base_hours = {
            'critical': 16,
            'high': 8,
            'medium': 4,
            'low': 2
        }
        
        multipliers = {
            'removal': 2.0,
            'schema': 1.5,
            'logic': 1.2,
            'data': 1.0
        }
        
        return int(base_hours[severity] * multipliers.get(change_type, 1.0))
    
    def _generate_recommendations(self, change_type: str, severity_breakdown: Dict[str, int], 
                                downstream_nodes: Set[str]) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []
        
        total_critical = severity_breakdown['critical']
        total_high = severity_breakdown['high']
        total_affected = len(downstream_nodes)
        
        if total_critical > 0:
            recommendations.append(f"URGENT: {total_critical} critical systems affected - coordinate with all teams")
            recommendations.append("Schedule change during maintenance window")
            recommendations.append("Prepare rollback plan before deployment")
        
        if total_high > 0:
            recommendations.append(f"HIGH PRIORITY: {total_high} high-impact systems need attention")
            recommendations.append("Notify affected teams at least 48 hours in advance")
        
        if total_affected > 10:
            recommendations.append(f"LARGE IMPACT: {total_affected} systems affected - consider phased rollout")
        
        if change_type == 'removal':
            recommendations.append("Provide migration guide for affected systems")
            recommendations.append("Consider deprecation period before removal")
        
        recommendations.append("Update all documentation and data contracts")
        recommendations.append("Monitor system health after deployment")
        
        return recommendations

class GDPRLineageTracker:
    """
    Solution 5: Complete GDPR compliance lineage tracker
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.pii_catalog = {}
        self.retention_policies = {}
        self.deletion_logs = []
    
    def register_pii_data(self, table: str, column: str, pii_type: str, 
                         retention_period: int) -> None:
        """Register PII data for GDPR tracking"""
        pii_key = f"{table}.{column}"
        
        self.pii_catalog[pii_key] = {
            'table': table,
            'column': column,
            'pii_type': pii_type,
            'retention_period': retention_period,
            'registered_at': datetime.now().isoformat(),
            'last_audited': None
        }
        
        self.retention_policies[pii_key] = {
            'retention_days': retention_period,
            'deletion_method': 'hard_delete',
            'backup_retention': 30,
            'compliance_notes': f"GDPR retention for {pii_type}"
        }
        
        logger.info(f"Registered PII: {pii_key} ({pii_type}) - {retention_period} days retention")
    
    def find_personal_data_usage(self, user_id: str) -> List[Dict[str, Any]]:
        """Find all systems that contain data for a specific user"""
        personal_data_locations = []
        
        # Start with known PII tables
        pii_tables = set()
        for pii_key in self.pii_catalog:
            table = self.pii_catalog[pii_key]['table']
            pii_tables.add(table)
        
        # For each PII table, find downstream systems
        for table in pii_tables:
            downstream_systems = self.lineage.get_downstream_lineage(table)
            
            for system in downstream_systems:
                if system in self.lineage.nodes:
                    node = self.lineage.nodes[system]
                    
                    # Check if this system likely contains user data
                    if self._contains_user_data(system, user_id):
                        location_info = {
                            'system': system,
                            'system_type': node.type,
                            'source_table': table,
                            'data_types': self._get_pii_types_in_system(system),
                            'retention_policy': self._get_retention_policy(system),
                            'deletion_complexity': self._assess_deletion_complexity(system),
                            'compliance_status': self._check_compliance_status(system)
                        }
                        
                        personal_data_locations.append(location_info)
        
        return personal_data_locations
    
    def generate_data_deletion_plan(self, user_id: str) -> Dict[str, Any]:
        """Generate plan for deleting all user data (Right to be Forgotten)"""
        locations = self.find_personal_data_usage(user_id)
        
        deletion_plan = {
            'user_id': user_id,
            'request_date': datetime.now().isoformat(),
            'systems_to_update': [],
            'deletion_order': [],
            'deletion_scripts': [],
            'verification_steps': [],
            'estimated_completion_time': 0,
            'compliance_checklist': []
        }
        
        # Sort by dependency order (delete downstream first)
        sorted_locations = self._sort_by_dependency(locations)
        
        total_time = 0
        for i, location in enumerate(sorted_locations):
            system = location['system']
            
            deletion_plan['systems_to_update'].append(location)
            deletion_plan['deletion_order'].append(system)
            
            # Generate deletion script
            script = self._generate_deletion_script(system, user_id, location)
            deletion_plan['deletion_scripts'].append(script)
            
            # Generate verification steps
            verification = self._generate_verification_steps(system, user_id)
            deletion_plan['verification_steps'].extend(verification)
            
            # Estimate time
            time_estimate = self._estimate_deletion_time(location)
            total_time += time_estimate
        
        deletion_plan['estimated_completion_time'] = total_time
        deletion_plan['compliance_checklist'] = self._generate_compliance_checklist(user_id, locations)
        
        return deletion_plan
    
    def _contains_user_data(self, system: str, user_id: str) -> bool:
        """Check if system contains data for specific user"""
        # In a real implementation, this would query the actual system
        # For this solution, we'll use heuristics based on system metadata
        
        if system not in self.lineage.nodes:
            return False
        
        node = self.lineage.nodes[system]
        metadata = node.metadata
        
        # Check if system has user_id column or similar
        user_columns = ['user_id', 'customer_id', 'account_id', 'email']
        system_columns = metadata.get('columns', [])
        
        return any(col in system_columns for col in user_columns)
    
    def _get_pii_types_in_system(self, system: str) -> List[str]:
        """Get types of PII data in system"""
        pii_types = []
        
        # Check upstream lineage for PII sources
        upstream_systems = self.lineage.get_upstream_lineage(system)
        
        for upstream in upstream_systems:
            for pii_key, pii_info in self.pii_catalog.items():
                if pii_info['table'] == upstream:
                    pii_types.append(pii_info['pii_type'])
        
        return list(set(pii_types))
    
    def _get_retention_policy(self, system: str) -> Dict[str, Any]:
        """Get data retention policy for system"""
        # Default retention policy
        default_policy = {
            'retention_days': 2555,  # 7 years
            'deletion_method': 'soft_delete',
            'backup_retention': 90,
            'policy_source': 'default'
        }
        
        # Check if system has specific retention policy
        for pii_key, policy in self.retention_policies.items():
            table = pii_key.split('.')[0]
            if table == system:
                return policy
        
        return default_policy
    
    def _assess_deletion_complexity(self, system: str) -> str:
        """Assess complexity of deleting data from system"""
        if system not in self.lineage.nodes:
            return 'unknown'
        
        node = self.lineage.nodes[system]
        downstream_count = len(self.lineage.get_downstream_lineage(system))
        
        if downstream_count > 10:
            return 'high'
        elif downstream_count > 3:
            return 'medium'
        else:
            return 'low'
    
    def _check_compliance_status(self, system: str) -> str:
        """Check GDPR compliance status of system"""
        # This would integrate with compliance monitoring systems
        # For this solution, we'll use simple heuristics
        
        if system not in self.lineage.nodes:
            return 'unknown'
        
        node = self.lineage.nodes[system]
        metadata = node.metadata
        
        # Check for compliance indicators
        has_encryption = metadata.get('encrypted', False)
        has_access_controls = metadata.get('access_controls', False)
        has_audit_log = metadata.get('audit_logging', False)
        
        compliance_score = sum([has_encryption, has_access_controls, has_audit_log])
        
        if compliance_score >= 3:
            return 'compliant'
        elif compliance_score >= 2:
            return 'partially_compliant'
        else:
            return 'non_compliant'
    
    def _sort_by_dependency(self, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort locations by dependency order (downstream first)"""
        # Create dependency graph
        dependency_graph = {}
        for location in locations:
            system = location['system']
            upstream = self.lineage.get_upstream_lineage(system)
            dependency_graph[system] = len(upstream)
        
        # Sort by dependency count (fewer dependencies first)
        return sorted(locations, key=lambda x: dependency_graph.get(x['system'], 0))
    
    def _generate_deletion_script(self, system: str, user_id: str, location: Dict[str, Any]) -> Dict[str, str]:
        """Generate deletion script for a system"""
        system_type = location.get('system_type', 'table')
        
        if system_type == 'table':
            sql_script = f"""
            -- Delete user data from {system}
            DELETE FROM {system} WHERE user_id = '{user_id}';
            
            -- Verify deletion
            SELECT COUNT(*) FROM {system} WHERE user_id = '{user_id}';
            -- Expected result: 0
            """
            
            return {
                'system': system,
                'script_type': 'sql',
                'script': sql_script,
                'execution_order': 1,
                'rollback_script': f"-- No rollback possible for GDPR deletion from {system}"
            }
        
        else:
            # For other system types, generate appropriate scripts
            return {
                'system': system,
                'script_type': 'manual',
                'script': f"Manual deletion required for {system} - contact system owner",
                'execution_order': 2,
                'rollback_script': "No rollback possible"
            }
    
    def _generate_verification_steps(self, system: str, user_id: str) -> List[str]:
        """Generate verification steps for deletion"""
        return [
            f"Verify no records exist for user {user_id} in {system}",
            f"Check backup systems for {system}",
            f"Verify downstream systems no longer receive data for user {user_id}",
            f"Update audit logs for {system} deletion"
        ]
    
    def _estimate_deletion_time(self, location: Dict[str, Any]) -> int:
        """Estimate time required for deletion (in minutes)"""
        complexity = location.get('deletion_complexity', 'medium')
        
        time_estimates = {
            'low': 15,
            'medium': 30,
            'high': 60
        }
        
        return time_estimates.get(complexity, 30)
    
    def _generate_compliance_checklist(self, user_id: str, locations: List[Dict[str, Any]]) -> List[str]:
        """Generate GDPR compliance checklist"""
        return [
            f"Verify identity of data subject requesting deletion for user {user_id}",
            "Document the deletion request and legal basis",
            f"Complete deletion from all {len(locations)} identified systems",
            "Verify deletion from backup systems",
            "Update data processing records",
            "Notify third parties if data was shared",
            "Document completion of deletion request",
            "Retain deletion logs for audit purposes"
        ]

class DataQualityLineageAnalyzer:
    """
    Solution 6: Complete data quality root cause analyzer
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.quality_metrics = {}
        self.quality_issues = []
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.95,
            'timeliness': 0.90,
            'validity': 0.95
        }
    
    def register_quality_issue(self, dataset: str, issue_type: str, 
                              issue_description: str, severity: str) -> None:
        """Register a data quality issue"""
        issue = {
            'id': f"issue_{len(self.quality_issues) + 1}",
            'dataset': dataset,
            'issue_type': issue_type,
            'description': issue_description,
            'severity': severity,
            'reported_at': datetime.now().isoformat(),
            'status': 'open',
            'root_cause_analysis': None
        }
        
        self.quality_issues.append(issue)
        logger.info(f"Registered quality issue: {issue_type} in {dataset} ({severity})")
    
    def analyze_quality_root_cause(self, dataset: str, issue_type: str) -> Dict[str, Any]:
        """Analyze root cause of data quality issue using lineage"""
        # Get upstream lineage to find potential root causes
        upstream_systems = self.lineage.get_upstream_lineage(dataset)
        
        analysis = {
            'dataset': dataset,
            'issue_type': issue_type,
            'analysis_timestamp': datetime.now().isoformat(),
            'potential_root_causes': [],
            'affected_downstream_systems': [],
            'confidence_scores': {},
            'recommendations': []
        }
        
        # Analyze upstream systems for quality issues
        for upstream in upstream_systems:
            upstream_quality = self._get_quality_metrics(upstream)
            
            if self._has_quality_issues(upstream_quality, issue_type):
                confidence = self._calculate_confidence(upstream_quality, issue_type)
                
                root_cause = {
                    'system': upstream,
                    'quality_metrics': upstream_quality,
                    'confidence': confidence,
                    'issue_indicators': self._identify_issue_indicators(upstream_quality, issue_type),
                    'suggested_fixes': self._suggest_fixes(upstream, issue_type, upstream_quality)
                }
                
                analysis['potential_root_causes'].append(root_cause)
                analysis['confidence_scores'][upstream] = confidence
        
        # Get downstream systems that might be affected
        downstream_systems = self.lineage.get_downstream_lineage(dataset)
        for downstream in downstream_systems:
            if downstream in self.lineage.nodes:
                node = self.lineage.nodes[downstream]
                analysis['affected_downstream_systems'].append({
                    'system': downstream,
                    'type': node.type,
                    'impact_level': self._assess_downstream_impact(downstream, issue_type)
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_quality_recommendations(
            analysis['potential_root_causes'],
            analysis['affected_downstream_systems'],
            issue_type
        )
        
        return analysis
    
    def _get_quality_metrics(self, system: str) -> Dict[str, float]:
        """Get quality metrics for a system"""
        # In a real implementation, this would query quality monitoring systems
        # For this solution, we'll simulate quality metrics
        
        if system in self.quality_metrics:
            return self.quality_metrics[system]
        
        # Simulate quality metrics with some randomness
        import random
        random.seed(hash(system) % 1000)  # Deterministic randomness based on system name
        
        metrics = {
            'completeness': max(0.7, random.uniform(0.85, 1.0)),
            'accuracy': max(0.8, random.uniform(0.90, 1.0)),
            'consistency': max(0.75, random.uniform(0.88, 1.0)),
            'timeliness': max(0.7, random.uniform(0.85, 0.98)),
            'validity': max(0.8, random.uniform(0.90, 1.0))
        }
        
        self.quality_metrics[system] = metrics
        return metrics
    
    def _has_quality_issues(self, quality_metrics: Dict[str, float], target_issue: str) -> bool:
        """Check if quality metrics indicate issues similar to target issue"""
        threshold = self.quality_thresholds.get(target_issue, 0.95)
        
        if target_issue in quality_metrics:
            return quality_metrics[target_issue] < threshold
        
        # Check related metrics
        related_metrics = {
            'completeness': ['completeness'],
            'accuracy': ['accuracy', 'validity'],
            'consistency': ['consistency', 'accuracy'],
            'timeliness': ['timeliness'],
            'validity': ['validity', 'accuracy']
        }
        
        for metric in related_metrics.get(target_issue, [target_issue]):
            if metric in quality_metrics:
                if quality_metrics[metric] < self.quality_thresholds.get(metric, 0.95):
                    return True
        
        return False
    
    def _calculate_confidence(self, quality_metrics: Dict[str, float], target_issue: str) -> float:
        """Calculate confidence that this system is the root cause"""
        if target_issue not in quality_metrics:
            return 0.5
        
        score = quality_metrics[target_issue]
        threshold = self.quality_thresholds[target_issue]
        
        # Higher confidence for lower scores (further from threshold)
        if score < threshold:
            confidence = min(0.95, (threshold - score) / threshold)
        else:
            confidence = 0.1  # Low confidence if above threshold
        
        return round(confidence, 2)
    
    def _identify_issue_indicators(self, quality_metrics: Dict[str, float], issue_type: str) -> List[str]:
        """Identify specific indicators of quality issues"""
        indicators = []
        
        for metric, score in quality_metrics.items():
            threshold = self.quality_thresholds.get(metric, 0.95)
            if score < threshold:
                gap = threshold - score
                indicators.append(f"{metric}: {score:.2f} (below threshold {threshold}, gap: {gap:.2f})")
        
        return indicators
    
    def _suggest_fixes(self, system: str, issue_type: str, quality_metrics: Dict[str, float]) -> List[str]:
        """Suggest fixes for quality issues"""
        fixes = []
        
        fix_suggestions = {
            'completeness': [
                "Add data validation rules to reject incomplete records",
                "Implement default value handling for missing fields",
                "Set up monitoring for data source availability"
            ],
            'accuracy': [
                "Implement data validation rules at ingestion",
                "Add data quality checks in transformation pipeline",
                "Set up automated data profiling and anomaly detection"
            ],
            'consistency': [
                "Standardize data formats across sources",
                "Implement referential integrity checks",
                "Add data normalization steps in pipeline"
            ],
            'timeliness': [
                "Optimize data pipeline performance",
                "Set up SLA monitoring for data freshness",
                "Implement incremental data processing"
            ],
            'validity': [
                "Add schema validation at data ingestion",
                "Implement business rule validation",
                "Set up data type and format checks"
            ]
        }
        
        # Add general fixes
        fixes.extend(fix_suggestions.get(issue_type, []))
        
        # Add specific fixes based on quality scores
        for metric, score in quality_metrics.items():
            if score < self.quality_thresholds.get(metric, 0.95):
                fixes.extend(fix_suggestions.get(metric, []))
        
        return list(set(fixes))  # Remove duplicates
    
    def _assess_downstream_impact(self, system: str, issue_type: str) -> str:
        """Assess impact level on downstream system"""
        if system not in self.lineage.nodes:
            return 'unknown'
        
        node = self.lineage.nodes[system]
        metadata = node.metadata
        
        # High impact for production systems
        if metadata.get('environment') == 'production':
            return 'high'
        elif metadata.get('environment') in ['staging', 'analytics']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_quality_recommendations(self, root_causes: List[Dict[str, Any]], 
                                        downstream_systems: List[Dict[str, Any]], 
                                        issue_type: str) -> List[str]:
        """Generate recommendations based on root cause analysis"""
        recommendations = []
        
        if not root_causes:
            recommendations.append("No clear root cause identified - investigate data sources manually")
            recommendations.append(f"Set up monitoring for {issue_type} issues")
        else:
            # Sort by confidence
            sorted_causes = sorted(root_causes, key=lambda x: x['confidence'], reverse=True)
            top_cause = sorted_causes[0]
            
            recommendations.append(
                f"Investigate {top_cause['system']} - highest confidence root cause ({top_cause['confidence']:.0%})"
            )
            
            # Add specific fixes from top cause
            recommendations.extend(top_cause['suggested_fixes'][:3])  # Top 3 fixes
        
        # Add downstream impact recommendations
        high_impact_systems = [s for s in downstream_systems if s['impact_level'] == 'high']
        if high_impact_systems:
            recommendations.append(
                f"URGENT: {len(high_impact_systems)} high-impact systems affected - notify teams immediately"
            )
        
        # Add monitoring recommendations
        recommendations.extend([
            f"Set up automated monitoring for {issue_type} issues",
            "Implement data quality alerts and notifications",
            "Create data quality dashboard for ongoing monitoring"
        ])
        
        return recommendations

class LineageExerciseRunner:
    """Complete exercise runner with all implementations"""
    
    def __init__(self):
        self.sql_extractor = SQLLineageExtractor()
        self.lineage_tracker = MultiHopLineageTracker()
        self.column_tracker = ColumnLevelLineageTracker()
        self.impact_analyzer = ImpactAnalyzer(self.lineage_tracker)
        self.gdpr_tracker = GDPRLineageTracker(self.lineage_tracker)
        self.quality_analyzer = DataQualityLineageAnalyzer(self.lineage_tracker)
        
        # Set up sample data
        self._setup_sample_data()
    
    def _setup_sample_data(self):
        """Set up sample data for demonstrations"""
        # Add sample nodes
        self.lineage_tracker.add_node('raw.users', 'table', {
            'schema': 'raw',
            'columns': ['user_id', 'email', 'name', 'created_at'],
            'environment': 'production'
        })
        
        self.lineage_tracker.add_node('raw.orders', 'table', {
            'schema': 'raw',
            'columns': ['order_id', 'user_id', 'total_amount', 'created_at'],
            'environment': 'production'
        })
        
        self.lineage_tracker.add_node('analytics.user_metrics', 'table', {
            'schema': 'analytics',
            'columns': ['user_id', 'email', 'order_count', 'total_spent'],
            'environment': 'production'
        })
        
        # Add sample edges
        self.lineage_tracker.add_edge(
            'raw.users', 'analytics.user_metrics',
            'join_aggregation', 'JOIN and GROUP BY user_id'
        )
        
        self.lineage_tracker.add_edge(
            'raw.orders', 'analytics.user_metrics',
            'aggregation', 'SUM(total_amount), COUNT(order_id)'
        )
        
        # Add sample column mappings
        self.column_tracker.add_column_mapping(
            'raw.users', 'user_id', 'analytics.user_metrics', 'user_id',
            'direct', 'u.user_id'
        )
        
        self.column_tracker.add_column_mapping(
            'raw.users', 'email', 'analytics.user_metrics', 'email',
            'direct', 'u.email'
        )
        
        # Register PII data
        self.gdpr_tracker.register_pii_data('raw.users', 'email', 'email', 2555)
        self.gdpr_tracker.register_pii_data('raw.users', 'name', 'name', 2555)
    
    def run_exercise_1_sql_parsing(self):
        """Run Exercise 1: SQL parsing for lineage extraction"""
        print("=== Exercise 1: SQL Parsing for Lineage Extraction ===")
        
        test_queries = [
            """
            INSERT INTO analytics.user_metrics
            SELECT 
                u.user_id,
                u.email,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM ecommerce.users u
            LEFT JOIN ecommerce.orders o ON u.user_id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.user_id, u.email
            """,
            """
            WITH monthly_sales AS (
                SELECT 
                    DATE_TRUNC('month', created_at) as month,
                    SUM(total_amount) as monthly_revenue
                FROM orders
                WHERE status = 'completed'
                GROUP BY DATE_TRUNC('month', created_at)
            )
            SELECT * FROM monthly_sales
            WHERE monthly_revenue > 10000
            """
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print("Table Lineage:", self.sql_extractor.extract_table_lineage(query))
            print("Column Lineage:", self.sql_extractor.extract_column_lineage(query))
            complex_analysis = self.sql_extractor.parse_complex_query(query)
            print(f"Complexity Score: {complex_analysis['complexity_score']}")
    
    def run_exercise_2_multi_hop_lineage(self):
        """Run Exercise 2: Multi-hop lineage tracking"""
        print("\n=== Exercise 2: Multi-hop Lineage Tracking ===")
        
        # Test upstream lineage
        upstream = self.lineage_tracker.get_upstream_lineage('analytics.user_metrics')
        print(f"Upstream of analytics.user_metrics: {upstream}")
        
        # Test downstream lineage
        downstream = self.lineage_tracker.get_downstream_lineage('raw.users')
        print(f"Downstream of raw.users: {downstream}")
        
        # Test shortest path
        path = self.lineage_tracker.find_shortest_path('raw.users', 'analytics.user_metrics')
        print(f"Shortest path: {' -> '.join(path)}")
        
        # Test subgraph
        subgraph = self.lineage_tracker.get_lineage_subgraph('analytics.user_metrics')
        print(f"Subgraph around analytics.user_metrics: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    
    def run_exercise_3_column_lineage(self):
        """Run Exercise 3: Column-level lineage tracking"""
        print("\n=== Exercise 3: Column-level Lineage Tracking ===")
        
        # Test column lineage tracing
        lineage_chain = self.column_tracker.trace_column_lineage('analytics.user_metrics', 'email')
        print(f"Email column lineage: {len(lineage_chain)} steps")
        for step in lineage_chain:
            print(f"  Step {step['step']}: {step['source_table']}.{step['source_column']} -> {step['target_table']}.{step['target_column']} ({step['transformation_type']})")
        
        # Test PII lineage detection
        pii_lineage = self.column_tracker.find_pii_lineage()
        print(f"\nFound {len(pii_lineage)} PII lineage mappings:")
        for pii in pii_lineage:
            print(f"  {pii['pii_type']}: {pii['source_table']}.{pii['source_column']} -> {pii['target_table']}.{pii['target_column']} (Risk: {pii['risk_level']})")
    
    def run_exercise_4_impact_analysis(self):
        """Run Exercise 4: Impact analysis"""
        print("\n=== Exercise 4: Impact Analysis ===")
        
        # Test change impact analysis
        impact = self.impact_analyzer.analyze_change_impact('raw.users', 'schema')
        print(f"\nSchema change impact on raw.users:")
        print(f"  Affected systems: {len(impact['affected_systems'])}")
        print(f"  Severity breakdown: {impact['severity_breakdown']}")
        print(f"  Estimated effort: {impact['estimated_effort_hours']} hours")
        print("  Top recommendations:")
        for rec in impact['recommendations'][:3]:
            print(f"    - {rec}")
    
    def run_exercise_5_gdpr_compliance(self):
        """Run Exercise 5: GDPR compliance tracking"""
        print("\n=== Exercise 5: GDPR Compliance Tracking ===")
        
        # Test personal data discovery
        user_data = self.gdpr_tracker.find_personal_data_usage('user_12345')
        print(f"\nPersonal data locations for user_12345: {len(user_data)} systems")
        for location in user_data:
            print(f"  {location['system']}: {location['data_types']} (Complexity: {location['deletion_complexity']})")
        
        # Test deletion plan generation
        deletion_plan = self.gdpr_tracker.generate_data_deletion_plan('user_12345')
        print(f"\nDeletion plan:")
        print(f"  Systems to update: {len(deletion_plan['systems_to_update'])}")
        print(f"  Estimated time: {deletion_plan['estimated_completion_time']} minutes")
        print(f"  Compliance checklist: {len(deletion_plan['compliance_checklist'])} items")
    
    def run_exercise_6_quality_analysis(self):
        """Run Exercise 6: Data quality root cause analysis"""
        print("\n=== Exercise 6: Data Quality Root Cause Analysis ===")
        
        # Register a quality issue
        self.quality_analyzer.register_quality_issue(
            'analytics.user_metrics', 'completeness',
            'Missing email addresses in 15% of records', 'high'
        )
        
        # Analyze root cause
        analysis = self.quality_analyzer.analyze_quality_root_cause(
            'analytics.user_metrics', 'completeness'
        )
        
        print(f"\nRoot cause analysis for completeness issue:")
        print(f"  Potential root causes: {len(analysis['potential_root_causes'])}")
        for cause in analysis['potential_root_causes']:
            print(f"    {cause['system']}: {cause['confidence']:.0%} confidence")
        
        print(f"  Affected downstream: {len(analysis['affected_downstream_systems'])}")
        print("  Top recommendations:")
        for rec in analysis['recommendations'][:3]:
            print(f"    - {rec}")
    
    def run_all_exercises(self):
        """Run all lineage tracking exercises"""
        print("🚀 Starting Data Lineage Tracking Exercises")
        print("=" * 60)
        
        self.run_exercise_1_sql_parsing()
        self.run_exercise_2_multi_hop_lineage()
        self.run_exercise_3_column_lineage()
        self.run_exercise_4_impact_analysis()
        self.run_exercise_5_gdpr_compliance()
        self.run_exercise_6_quality_analysis()
        
        print("\n✅ All exercises completed successfully!")
        print("\n🎯 Key Achievements:")
        print("  ✓ SQL parsing for automated lineage extraction")
        print("  ✓ Multi-hop lineage tracking with graph algorithms")
        print("  ✓ Column-level lineage with PII detection")
        print("  ✓ Impact analysis for change management")
        print("  ✓ GDPR compliance with data deletion planning")
        print("  ✓ Data quality root cause analysis")
        
        print("\n🚀 Next Steps:")
        print("  1. Integrate with real data catalog systems (DataHub, Atlas)")
        print("  2. Add real-time lineage updates from streaming systems")
        print("  3. Implement advanced visualization dashboards")
        print("  4. Set up monitoring and alerting for lineage health")
        print("  5. Deploy to production with proper security and scaling")

def main():
    """Run the complete data lineage tracking solution"""
    runner = LineageExerciseRunner()
    runner.run_all_exercises()

if __name__ == "__main__":
    main()