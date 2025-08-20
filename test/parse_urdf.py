#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""URDF文件解析工具 - 简化版"""

import xml.etree.ElementTree as ET
import os

def parse_urdf_simple(urdf_file):
    """简化解析URDF文件"""
    
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    # 1. 所有link组成的list
    links = []
    for link in root.findall('link'):
        links.append(link.get('name'))
    
    # 2. 所有joint组成的list
    joints = []
    for joint in root.findall('joint'):
        joints.append(joint.get('name'))
    
    # 3. 完整的运动学链关系 - 包括link到joint和joint到joint的关系
    joint_relations = []
    joint_dict = {}  # 存储每个joint的详细信息
    
    # 收集所有joint信息
    for joint in root.findall('joint'):
        name = joint.get('name')
        joint_type = joint.get('type', 'unknown')
        
        # 获取变换信息
        origin = joint.find('origin')
        xyz = origin.get('xyz', "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get('rpy', "0 0 0") if origin is not None else "0 0 0"
        
        # 获取父子链接
        parent = joint.find('parent')
        child = joint.find('child')
        parent_link = parent.get('link') if parent is not None else None
        child_link = child.get('link') if child is not None else None
        
        # 获取旋转轴信息
        axis = joint.find('axis')
        axis_xyz = axis.get('xyz', "0 0 0") if axis is not None else "0 0 0"
        
        # 获取关节限制信息
        limit = joint.find('limit')
        if limit is not None:
            lower = limit.get('lower', 'N/A')
            upper = limit.get('upper', 'N/A')
            effort = limit.get('effort', 'N/A')
            velocity = limit.get('velocity', 'N/A')
        else:
            lower = upper = effort = velocity = 'N/A'
        
        joint_dict[name] = {
            'type': joint_type,
            'parent_link': parent_link,
            'child_link': child_link,
            'xyz': xyz,
            'rpy': rpy,
            'axis': axis_xyz,
            'limits': {
                'lower': lower,
                'upper': upper,
                'effort': effort,
                'velocity': velocity
            }
        }
    
    # 建立完整的运动学链关系
    for joint_name, joint_info in joint_dict.items():
        parent_link = joint_info['parent_link']
        child_link = joint_info['child_link']
        
        # 1. 查找从parent_link连接到当前joint的关系（如果parent_link是某个joint的child）
        for other_joint_name, other_joint_info in joint_dict.items():
            if other_joint_info['child_link'] == parent_link and joint_name != other_joint_name:
                relation = {
                    'from_joint': other_joint_name,
                    'to_joint': joint_name,
                    'type': joint_info['type'],
                    'xyz': joint_info['xyz'],
                    'rpy': joint_info['rpy'],
                    'axis': joint_info['axis'],
                    'limits': joint_info['limits']
                }
                joint_relations.append(relation)
                break
        else:
            # 如果没找到，说明parent_link是base link
            relation = {
                'from_joint': f"base({parent_link})",
                'to_joint': joint_name,
                'type': joint_info['type'],
                'xyz': joint_info['xyz'],
                'rpy': joint_info['rpy'],
                'axis': joint_info['axis'],
                'limits': joint_info['limits']
            }
            joint_relations.append(relation)
        
        # 2. 查找从当前joint连接到child_link的关系（如果child_link不是另一个joint的parent）
        has_child_joint = False
        for other_joint_name, other_joint_info in joint_dict.items():
            if other_joint_info['parent_link'] == child_link and joint_name != other_joint_name:
                has_child_joint = True
                break
        
        if not has_child_joint:
            # child_link是末端连杆
            relation = {
                'from_joint': joint_name,
                'to_joint': f"end({child_link})",
                'type': 'end_link',
                'xyz': "0 0 0",  # 末端连杆本身没有额外偏移
                'rpy': "0 0 0",
                'axis': "0 0 0",
                'limits': {
                    'lower': 'N/A',
                    'upper': 'N/A', 
                    'effort': 'N/A',
                    'velocity': 'N/A'
                }
            }
            joint_relations.append(relation)
    
    return links, joints, joint_relations

if __name__ == "__main__":
    urdf_file = os.path.join(os.path.dirname(__file__), "..", "resources", "urdf", "so100_tcp.urdf")
    
    if not os.path.exists(urdf_file):
        print(f"错误: URDF文件不存在 {urdf_file}")
        exit(1)
    
    links, joints, joint_relations = parse_urdf_simple(urdf_file)
    
    print("URDF解析结果:")
    print("="*50)
    
    # 1. 所有link
    print(f"\n1. Links ({len(links)}个):")
    print(links)
    
    # 2. 所有joint
    print(f"\n2. Joints ({len(joints)}个):")
    print(joints)
    
    # 3. 完整的运动学链关系
    print(f"\n3. 完整运动学链关系 ({len(joint_relations)}个):")
    for i, relation in enumerate(joint_relations, 1):
        print(f"  {i}. {relation['from_joint']} -> {relation['to_joint']}")
        print(f"     类型: {relation['type']}")
        print(f"     位置: {relation['xyz']}")
        print(f"     旋转: {relation['rpy']}")
        print(f"     旋转轴: {relation['axis']}")
        print(f"     限制: 下限={relation['limits']['lower']}, 上限={relation['limits']['upper']}")
        print(f"          力矩={relation['limits']['effort']}, 速度={relation['limits']['velocity']}")
        print()
