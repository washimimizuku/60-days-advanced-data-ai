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