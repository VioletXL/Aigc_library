#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图书推荐系统 - 混合推荐算法（注意力机制+院系协同）"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

DECAY_DAYS = 120
ALPHA = 1.0
REPEAT_BOOST = 0.2
AUTHOR_BOOST = 0.5
DEPT_BOOST = 0.275
PRESS_BOOST = 0.175
CATEGORY2_BOOST = 0.15
TIME_PATTERN_BOOST = 0.10
AUTHOR_PRESS_BOOST = 0.08
SEQUENCE_BOOST = 0.12
HIGH_FREQUENCY_THRESHOLD = 5
COLLABORATIVE_BOOST = 0.1
BOOK_SIMILARITY_BOOST = 0.1
LIGHTFM_WEIGHT = 0.45
LIGHTFM_TOPK = 50
ATTENTION_TEMPERATURE = 0.05
DEPT_COLLABORATIVE_BOOST = 0.3
DEPT_AFFINITY_BOOST = 0.25


class HybridRecommender:

    def __init__(self):
        self.user_history = defaultdict(list)
        self.author_pref = defaultdict(lambda: defaultdict(float))
        self.press_pref = defaultdict(lambda: defaultdict(float))
        self.dept_pref = defaultdict(lambda: defaultdict(float))
        self.category1_pref = defaultdict(lambda: defaultdict(float))
        self.category2_pref = defaultdict(lambda: defaultdict(float))
        self.author_press_pref = defaultdict(lambda: defaultdict(float))
        self.time_pattern = defaultdict(lambda: defaultdict(float))
        self.book_features = {}
        self.user_info = {}
        self.stats = {
            'user_borrow_counts': defaultdict(int),
            'book_borrow_counts': defaultdict(int),
            'time_distribution': defaultdict(int),
            'author_popularity': defaultdict(int),
            'category_distribution': defaultdict(int),
            'dept_distribution': defaultdict(int),
        }
        self.sequence_patterns = defaultdict(lambda: defaultdict(int))
        self.user_similarity = defaultdict(list)
        self.book_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        self.attention_weights = {}
        self.dept_book_affinity = defaultdict(lambda: defaultdict(float))
        self.dept_similarity = {}
        self.dept_users = defaultdict(list)
        
        self.author_to_books = defaultdict(list)
        self.category1_to_books = defaultdict(list)
        self.category2_to_books = defaultdict(list)
        self.global_popular_books = []
        self.dept_popular_books = defaultdict(list)
        
        self._title_index_ready = False
        self._tfidf_matrix = None
        self._title_bids = []
        self._title_nn_model = None
        
        self.lightfm_enabled = False
        self.lightfm_user_embeddings = None
        self.lightfm_item_embeddings = None
        self.lightfm_user_biases = None
        self.lightfm_item_biases = None
        self.lightfm_user_index = {}
        self.lightfm_item_index = {}
        self.lightfm_item_ids = []
        
        self.book_id_to_str = {}
        self.str_to_book_id = {}
        self.user_id_to_str = {}
        self.str_to_user_id = {}

    def load_data(self):
        # 获取项目根目录（code目录的上一级）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        
        interactions = pd.read_csv(os.path.join(data_dir, 'inter_reevaluation.csv'))
        interactions.rename(columns={'借阅时间': 'borrow_time'}, inplace=True)
        interactions['borrow_time'] = pd.to_datetime(interactions['borrow_time'])
        interactions['borrow_month'] = interactions['borrow_time'].dt.month
        interactions['borrow_dayofweek'] = interactions['borrow_time'].dt.dayofweek
        interactions['borrow_hour'] = interactions['borrow_time'].dt.hour

        items = pd.read_csv(os.path.join(data_dir, 'item.csv'))
        for _, row in items.iterrows():
            bid = row['book_id']
            author = row['作者']
            cat1 = row['一级分类']
            cat2 = row['二级分类']

            self.book_features[bid] = {
                'author': author,
                'press': row['出版社'],
                'category1': cat1,
                'category2': cat2,
                'title': row['题名']
            }

            bid_str = self._id_to_str(bid)
            self.book_id_to_str[bid] = bid_str
            self.str_to_book_id.setdefault(bid_str, bid)

            if pd.notna(author):
                self.author_to_books[author].append(bid)
            if pd.notna(cat1):
                self.category1_to_books[cat1].append(bid)
            if pd.notna(cat2):
                self.category2_to_books[cat2].append(bid)

            self.stats['author_popularity'][author] += 1
            if pd.notna(cat1):
                self.stats['category_distribution'][cat1] += 1

        users = pd.read_csv(os.path.join(data_dir, 'user.csv'))
        for _, row in users.iterrows():
            uid = row['借阅人']
            dept = row['DEPT']
            self.user_info[uid] = {
                'dept': dept,
                'gender': row['性别'],
                'grade': row['年级']
            }
            self.stats['dept_distribution'][dept] += 1

            uid_str = self._id_to_str(uid)
            self.user_id_to_str[uid] = uid_str
            self.str_to_user_id.setdefault(uid_str, uid)

        return interactions

    @staticmethod
    def _id_to_str(value):
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        return str(value)

    def _book_id_from_str(self, bid_str):
        if bid_str in self.str_to_book_id:
            return self.str_to_book_id[bid_str]
        try:
            if isinstance(bid_str, str):
                if '.' in bid_str:
                    candidate = int(float(bid_str))
                else:
                    candidate = int(bid_str)
                if candidate in self.book_features:
                    return candidate
        except ValueError:
            pass
        return bid_str

    def _user_id_from_str(self, uid_str):
        if uid_str in self.str_to_user_id:
            return self.str_to_user_id[uid_str]
        try:
            if isinstance(uid_str, str):
                candidate = int(uid_str)
                if candidate in self.user_info:
                    return candidate
        except ValueError:
            pass
        return uid_str

    def recommend(self, user_id, topk=1):
        """
        为指定用户生成图书推荐（支持历史 + 新书候选）

        策略：
        1. 生成多源候选：历史/全局热门/作者分类相似/标题相似
        2. 基于用户偏好与全局特征打分
        3. 返回得分最高的图书

        Args:
            user_id: 用户 ID
            topk: 需要的推荐数量（当前用于未来扩展，默认返回Top-1）

        Returns:
            str: 推荐的图书 ID，如无候选则返回 None
        """
        history = self.user_history.get(user_id, [])
        if not history:
            popular = self._popular_books(topk=topk)
            return popular[0] if popular else None

        candidates = self.get_candidates(user_id)
        if not candidates:
            candidates = [book_id for book_id, _, _ in history]

        borrow_count = defaultdict(int)
        for book_id, _, _ in history:
            borrow_count[book_id] += 1

        user_dept = self.user_info.get(user_id, {}).get('dept', None)
        scores = {}
        history_books = set(borrow_count.keys())

        for book_id in candidates:
            if book_id not in self.book_features:
                continue

            feats = self.book_features[book_id]
            author = feats.get('author')
            press = feats.get('press')
            category1 = feats.get('category1')
            category2 = feats.get('category2')

            repeat_count = borrow_count.get(book_id, 0)
            base_score = 0.0

            if repeat_count > 0:
                for hist_book, time_score, _ in history:
                    if hist_book == book_id:
                        base_score += time_score
            else:
                pref_score = 0.0
                pref_score += self.author_pref[user_id].get(author, 0.0)
                pref_score += 0.7 * self.press_pref[user_id].get(press, 0.0)
                if pd.notna(category1):
                    pref_score += 0.6 * self.category1_pref[user_id].get(category1, 0.0)
                if pd.notna(category2):
                    pref_score += 0.4 * self.category2_pref[user_id].get(category2, 0.0)
                pref_score += 0.05 * self.stats['book_borrow_counts'].get(book_id, 0)
                base_score = max(pref_score, 0.1)

            if base_score == 0:
                base_score = 0.05

            score = base_score

            if repeat_count > 1:
                multiplier = 1 + REPEAT_BOOST * (repeat_count - 1)
                if repeat_count >= HIGH_FREQUENCY_THRESHOLD:
                    multiplier *= 1.5
                score *= multiplier

            author_total = sum(self.author_pref[user_id].values())
            if author_total > 0 and author in self.author_pref[user_id]:
                author_ratio = self.author_pref[user_id][author] / author_total
                score *= (1 + AUTHOR_BOOST * author_ratio)

            press_total = sum(self.press_pref[user_id].values())
            if press_total > 0 and press in self.press_pref[user_id]:
                press_ratio = self.press_pref[user_id][press] / press_total
                score *= (1 + PRESS_BOOST * press_ratio)

            if user_dept and category1 in self.dept_pref.get(user_dept, {}):
                dept_total = sum(self.dept_pref[user_dept].values())
                if dept_total > 0:
                    dept_ratio = self.dept_pref[user_dept][category1] / dept_total
                    score *= (1 + DEPT_BOOST * dept_ratio)

            if pd.notna(category2) and category2 in self.category2_pref[user_id]:
                cat2_total = sum(self.category2_pref[user_id].values())
                if cat2_total > 0:
                    cat2_ratio = self.category2_pref[user_id][category2] / cat2_total
                    score *= (1 + CATEGORY2_BOOST * cat2_ratio)

            combo_key = f"{author}|{press}"
            combo_total = sum(self.author_press_pref[user_id].values())
            if combo_total > 0 and combo_key in self.author_press_pref[user_id]:
                combo_ratio = self.author_press_pref[user_id][combo_key] / combo_total
                score *= (1 + AUTHOR_PRESS_BOOST * combo_ratio)

            if repeat_count > 0 and user_id in self.sequence_patterns:
                sequence_boost = 0.0
                for pattern, count in self.sequence_patterns[user_id].items():
                    if pattern.endswith(f"→{book_id}"):
                        prev_book = pattern.split('→')[0]
                        if prev_book in history_books:
                            sequence_boost += count * SEQUENCE_BOOST
                if sequence_boost > 0:
                    score *= (1 + min(sequence_boost, 0.5))

            stability = 1.0
            if repeat_count >= 5:
                stability *= 1.3
            elif repeat_count >= 3:
                stability *= 1.15

            if repeat_count > 0:
                recent_borrow_times = [t for b, s, t in history if b == book_id]
                if recent_borrow_times:
                    days_since_last = (history[-1][2] - max(recent_borrow_times)).days
                    if days_since_last <= 30:
                        stability *= 1.25
                    elif days_since_last <= 90:
                        stability *= 1.1
            else:
                global_pop = self.stats['book_borrow_counts'].get(book_id, 0)
                if global_pop > 0:
                    stability *= 1 + min(global_pop / 500.0, 0.5)

            consistency_count = 0
            if author in self.author_pref[user_id]:
                consistency_count += 1
            if press in self.press_pref[user_id]:
                consistency_count += 1
            if pd.notna(category1) and category1 in self.category1_pref[user_id]:
                consistency_count += 1
            if consistency_count == 3:
                stability *= 1.2
            elif consistency_count == 2:
                stability *= 1.1

            score *= stability
            
            if repeat_count > 0 and user_id in self.attention_weights:
                attention_weight = self.attention_weights[user_id].get(book_id, 0.0)
                if attention_weight > 0:
                    score *= (1 + attention_weight * 0.3)  # 最高30%加成
            
            if user_dept and user_dept in self.dept_book_affinity:
                dept_affinity = self.dept_book_affinity[user_dept].get(book_id, 0.0)
                if dept_affinity > 0:
                    max_affinity = max(self.dept_book_affinity[user_dept].values())
                    if max_affinity > 0:
                        norm_affinity = dept_affinity / max_affinity
                        score *= (1 + DEPT_AFFINITY_BOOST * norm_affinity)
            
            if user_dept and user_dept in self.dept_similarity:
                dept_collab_score = 0.0
                for similar_dept, similarity in self.dept_similarity[user_dept][:3]:
                    if similar_dept in self.dept_book_affinity:
                        similar_affinity = self.dept_book_affinity[similar_dept].get(book_id, 0.0)
                        dept_collab_score += similarity * similar_affinity
                if dept_collab_score > 0:
                    score *= (1 + DEPT_COLLABORATIVE_BOOST * min(dept_collab_score / 10.0, 1.0))

            if repeat_count > 0 and history:
                last_book = history[-1][0]
                if book_id == last_book:
                    score *= (1 + ALPHA)
                else:
                    recent_books = [h[0] for h in history[-3:]]
                    if book_id in recent_books:
                        score *= 1.3

            if self.lightfm_enabled:
                lf_score = self._lightfm_score(user_id, book_id)
                if lf_score is not None:
                    lf_prob = 1.0 / (1.0 + np.exp(-lf_score))
                    score += LIGHTFM_WEIGHT * lf_prob

            score += 1e-6 * self.stats['book_borrow_counts'].get(book_id, 0)
            scores[book_id] = score

        if not scores:
            popular = self._popular_books(topk=topk)
            return popular[0] if popular else None

        return max(scores.items(), key=lambda x: x[1])[0]

    def build_features(self, interactions):
        """
        构建用户特征和偏好模型
        
        核心步骤：
        1. 计算时间衰减分数：越近期的借阅权重越高
        2. 累积用户对各维度特征的偏好（作者、出版社、分类等）
        3. 提取时间模式（月份、星期、小时）
        4. 构建序列模式（连续借阅关系）
        
        Args:
            interactions: 交互数据DataFrame
        """
        print("🔧 构建特征模型...")
        
        reference_time = interactions['borrow_time'].max()
        user_sequences = defaultdict(list)  # 用于存储时间序列
        
        for idx, row in interactions.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            borrow_time = row['borrow_time']
            borrow_month = row['borrow_month']
            borrow_dayofweek = row['borrow_dayofweek']
            borrow_hour = row['borrow_hour']
            
            time_diff = (reference_time - borrow_time).days
            time_score = np.exp(-time_diff / DECAY_DAYS)
            
            self.user_history[user_id].append((book_id, time_score, borrow_time))
            user_sequences[user_id].append((book_id, borrow_time))
            
            self.stats['user_borrow_counts'][user_id] += 1
            self.stats['book_borrow_counts'][book_id] += 1
            self.stats['time_distribution'][borrow_hour] += 1
            
            if book_id in self.book_features:
                features = self.book_features[book_id]
                author = features['author']
                press = features['press']
                category1 = features['category1']
                category2 = features['category2']
                
                self.author_pref[user_id][author] += time_score
                
                self.press_pref[user_id][press] += time_score
                
                if pd.notna(category1):
                    self.category1_pref[user_id][category1] += time_score
                
                if pd.notna(category2):
                    self.category2_pref[user_id][category2] += time_score
                
                author_press_combo = f"{author}|{press}"
                self.author_press_pref[user_id][author_press_combo] += time_score
                
                self.time_pattern[user_id][f"month_{borrow_month}"] += time_score
                self.time_pattern[user_id][f"dayofweek_{borrow_dayofweek}"] += time_score
                self.time_pattern[user_id][f"hour_{borrow_hour}"] += time_score
                
                if user_id in self.user_info:
                    dept = self.user_info[user_id]['dept']
                    self.dept_pref[dept][category1] += time_score
        
        for user_id, info in self.user_info.items():
            dept = info.get('dept')
            if dept:
                self.dept_users[dept].append(user_id)
        
        print("   构建序列模式...")
        self.sequence_patterns = defaultdict(lambda: defaultdict(int))
        for user_id, sequence in user_sequences.items():
            # 按时间排序
            sequence.sort(key=lambda x: x[1])
            for i in range(len(sequence) - 1):
                book1, time1 = sequence[i]
                book2, time2 = sequence[i + 1]
                days_diff = (time2 - time1).days
                if 0 < days_diff <= 30:  # 30天内算作有关联
                    self.sequence_patterns[user_id][f"{book1}→{book2}"] += 1
        
        print(f"   ✓ 用户特征数: {len(self.user_history)}")
        print(f"   ✓ 序列模式数: {sum(len(v) for v in self.sequence_patterns.values())}")
        
        print("   构建协同过滤模型...")
        self._build_collaborative_filtering()
        
        print("   构建图书相似度模型...")
        self._build_book_similarity()
        
        self.build_dept_collaborative()

        self.global_popular_books = [
            bid for bid, _ in sorted(
                self.stats['book_borrow_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        print("   计算注意力权重...")
        for user_id in self.user_history.keys():
            self.attention_weights[user_id] = self.compute_attention_weights(user_id)
        
        print()
    
    def _build_collaborative_filtering(self):
        """构建基于用户的协同过滤模型"""
        from collections import Counter
        
        for user_id in self.user_history.keys():
            user_books = set([b for b, _, _ in self.user_history[user_id]])
            
            similarities = []
            for other_id in self.user_history.keys():
                if other_id == user_id:
                    continue
                other_books = set([b for b, _, _ in self.user_history[other_id]])
                
                intersection = len(user_books & other_books)
                union = len(user_books | other_books)
                
                if union > 0 and intersection >= 3:  # 至少有3本共同书籍
                    similarity = intersection / union
                    similarities.append((other_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.user_similarity[user_id] = similarities[:10]
    
    def _build_book_similarity(self):
        """构建图书共现矩阵（经常一起被借阅的书）"""
        for user_id, history in self.user_history.items():
            books = [b for b, _, _ in history]
            
            # 对于每对书籍，增加共现计数
            for i in range(len(books)):
                for j in range(i + 1, len(books)):
                    book1, book2 = books[i], books[j]
                    self.book_cooccurrence[book1][book2] += 1
                    self.book_cooccurrence[book2][book1] += 1

    def load_lightfm_model(self, model_path='output/lightfm_model.npz', mapping_path=None):
        # 如果传入的是相对路径，转换为绝对路径
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)
        
        if mapping_path is None:
            base, _ = os.path.splitext(model_path)
            mapping_path = f"{base}_mappings.json"
        elif not os.path.isabs(mapping_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mapping_path = os.path.join(project_root, mapping_path)

        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            print(f"   ⚠ 未找到 LightFM 模型或映射文件 ({model_path})，跳过融合。")
            self.lightfm_enabled = False
            return

        try:
            data = np.load(model_path)
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except Exception as err:
            print(f"   ⚠ 加载 LightFM 模型失败: {err}")
            self.lightfm_enabled = False
            return

        self.lightfm_user_embeddings = data.get('user_embeddings')
        self.lightfm_item_embeddings = data.get('item_embeddings')
        self.lightfm_user_biases = data.get('user_biases')
        self.lightfm_item_biases = data.get('item_biases')

        user_ids = [self._id_to_str(uid) for uid in mapping.get('user_ids', [])]
        item_ids = [self._id_to_str(iid) for iid in mapping.get('item_ids', [])]

        self.lightfm_user_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.lightfm_item_index = {iid: idx for idx, iid in enumerate(item_ids)}
        self.lightfm_item_ids = item_ids

        if self.lightfm_item_embeddings is None or self.lightfm_user_embeddings is None:
            print("   ⚠ LightFM 模型内容不完整，已跳过。")
            self.lightfm_enabled = False
            return

        self.lightfm_enabled = True
        print("   ✓ LightFM 模型已加载，启用协同分数融合。")

    def _lightfm_score(self, user_id, book_id):
        if not self.lightfm_enabled:
            return None
        user_key = self._id_to_str(user_id)
        item_key = self._id_to_str(book_id)
        u_idx = self.lightfm_user_index.get(user_key)
        i_idx = self.lightfm_item_index.get(item_key)
        if u_idx is None or i_idx is None:
            return None

        user_vec = self.lightfm_user_embeddings[u_idx]
        item_vec = self.lightfm_item_embeddings[i_idx]
        score = float(np.dot(user_vec, item_vec))
        if self.lightfm_user_biases is not None:
            score += float(self.lightfm_user_biases[u_idx])
        if self.lightfm_item_biases is not None:
            score += float(self.lightfm_item_biases[i_idx])
        return score

    def _lightfm_top_items(self, user_id, topk=LIGHTFM_TOPK):
        if (
            not self.lightfm_enabled
            or self.lightfm_item_embeddings is None
            or self.lightfm_user_embeddings is None
        ):
            return []
        user_key = self._id_to_str(user_id)
        u_idx = self.lightfm_user_index.get(user_key)
        if u_idx is None:
            return []

        user_vec = self.lightfm_user_embeddings[u_idx]
        item_scores = np.dot(self.lightfm_item_embeddings, user_vec)
        if self.lightfm_item_biases is not None:
            item_scores = item_scores + self.lightfm_item_biases
        if self.lightfm_user_biases is not None:
            item_scores = item_scores + self.lightfm_user_biases[u_idx]

        max_items = min(len(self.lightfm_item_ids), len(item_scores))
        if max_items <= 0:
            return []

        item_scores = item_scores[:max_items]
        topk = min(topk, max_items)
        if topk <= 0:
            return []

        top_indices = np.argpartition(-item_scores, topk - 1)[:topk]
        top_indices = top_indices[np.argsort(-item_scores[top_indices])]

        candidates = []
        for idx in top_indices:
            if idx >= len(self.lightfm_item_ids):
                continue
            bid = self._book_id_from_str(self.lightfm_item_ids[idx])
            if bid in self.book_features:
                candidates.append(bid)
        return candidates

    
    def compute_attention_weights(self, user_id):
        """
        计算用户历史借阅记录的注意力权重
        结合时间、内容相似度、院系偏好三个维度
        """
        history = self.user_history.get(user_id, [])
        if not history:
            return {}
        
        n = len(history)
        if n == 0:
            return {}
        
        time_scores = np.array([score for _, score, _ in history])  # 时间衰减分数
        
        # 内容注意力：基于最后一本书的特征相似度
        last_book = history[-1][0]
        content_scores = np.zeros(n)
        if last_book in self.book_features:
            last_feats = self.book_features[last_book]
            for i, (book_id, _, _) in enumerate(history):
                if book_id in self.book_features:
                    feats = self.book_features[book_id]
                    sim = 0.0
                    if feats['author'] == last_feats['author']:
                        sim += 0.4
                    if feats['category1'] == last_feats['category1']:
                        sim += 0.3
                    if feats.get('category2') == last_feats.get('category2'):
                        sim += 0.2
                    if feats['press'] == last_feats['press']:
                        sim += 0.1
                    content_scores[i] = sim
        
        dept_scores = np.zeros(n)
        user_dept = self.user_info.get(user_id, {}).get('dept')
        if user_dept and user_dept in self.dept_pref:
            dept_prefs = self.dept_pref[user_dept]
            total_dept_score = sum(dept_prefs.values())
            if total_dept_score > 0:
                for i, (book_id, _, _) in enumerate(history):
                    if book_id in self.book_features:
                        cat1 = self.book_features[book_id].get('category1')
                        if cat1 and cat1 in dept_prefs:
                            dept_scores[i] = dept_prefs[cat1] / total_dept_score
        
        # 融合三种注意力（加权平均）
        combined_scores = 0.5 * time_scores / (time_scores.sum() + 1e-9)
        combined_scores += 0.3 * content_scores / (content_scores.sum() + 1e-9)
        combined_scores += 0.2 * dept_scores / (dept_scores.sum() + 1e-9)
        
        # Softmax归一化（使用温度参数控制分布尖锐程度）
        exp_scores = np.exp(combined_scores / ATTENTION_TEMPERATURE)
        attention_weights = exp_scores / (exp_scores.sum() + 1e-9)
        
        # 存储权重
        weights = {}
        for i, (book_id, _, _) in enumerate(history):
            weights[book_id] = float(attention_weights[i])
        
        return weights
    
    def build_dept_collaborative(self):
        """
        构建院系协同过滤模型
        1. 计算院系-图书亲和度矩阵
        2. 计算院系间相似度
        3. 为每个院系生成热门书籍列表
        """
        print("   构建院系协同过滤...")
        
        # 1. 计算院系-图书亲和度（考虑时间衰减）
        for user_id, history in self.user_history.items():
            user_dept = self.user_info.get(user_id, {}).get('dept')
            if not user_dept:
                continue
            for book_id, time_score, _ in history:
                self.dept_book_affinity[user_dept][book_id] += time_score
        
        # 2. 计算院系间相似度（基于图书借阅模式的余弦相似度）
        depts = list(self.dept_book_affinity.keys())
        n_depts = len(depts)
        
        if n_depts > 1:
            all_books = set()
            for dept_books in self.dept_book_affinity.values():
                all_books.update(dept_books.keys())
            all_books = list(all_books)
            book_to_idx = {b: i for i, b in enumerate(all_books)}
            
            dept_vectors = np.zeros((n_depts, len(all_books)))
            for i, dept in enumerate(depts):
                for book_id, score in self.dept_book_affinity[dept].items():
                    j = book_to_idx[book_id]
                    dept_vectors[i, j] = score
            
            norms = np.linalg.norm(dept_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            dept_vectors = dept_vectors / norms
            
            similarity_matrix = np.dot(dept_vectors, dept_vectors.T)
            
            for i, dept_i in enumerate(depts):
                similar_depts = []
                for j, dept_j in enumerate(depts):
                    if i != j:
                        similar_depts.append((dept_j, similarity_matrix[i, j]))
                # 按相似度排序，保留top-5
                similar_depts.sort(key=lambda x: x[1], reverse=True)
                self.dept_similarity[dept_i] = similar_depts[:5]
        
        # 3. 为每个院系生成热门书籍列表
        for dept, books in self.dept_book_affinity.items():
            sorted_books = sorted(books.items(), key=lambda x: x[1], reverse=True)
            self.dept_popular_books[dept] = [b[0] for b in sorted_books[:100]]
        
        print(f"      ✓ 院系数: {n_depts}, 平均每院系热门书籍: {np.mean([len(b) for b in self.dept_popular_books.values()]):.1f}")

    def _ensure_title_index(self, n_neighbors=20):
        """懒构建基于标题的 TF-IDF 最近邻索引"""
        if self._title_index_ready:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors

        titles = []
        bids = []
        for bid, feats in self.book_features.items():
            title = feats.get('title', '')
            titles.append(title if isinstance(title, str) else '')
            bids.append(bid)

        if not bids:
            self._title_index_ready = True
            return

        vectorizer = TfidfVectorizer(max_features=5000)
        matrix = vectorizer.fit_transform(titles)
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn_model.fit(matrix)

        self._tfidf_matrix = matrix
        self._title_bids = bids
        self._title_nn_model = nn_model
        self._title_index_ready = True

    def _title_similar_books(self, book_id, topk=5):
        """基于标题的相似图书"""
        if book_id not in self.book_features:
            return []
        self._ensure_title_index()
        if not self._title_index_ready or self._title_nn_model is None:
            return []
        try:
            idx = self._title_bids.index(book_id)
        except ValueError:
            return []

        distances, indices = self._title_nn_model.kneighbors(self._tfidf_matrix[idx], n_neighbors=topk + 1)
        similar = []
        for ind in indices[0]:
            candidate = self._title_bids[ind]
            if candidate != book_id:
                similar.append(candidate)
        return similar[:topk]

    def _popular_books(self, topk=200):
        """获取全局热门书籍"""
        if not self.global_popular_books:
            return []
        return self.global_popular_books[:topk]

    def _similar_by_meta(self, book_id, topk=10):
        """基于作者与分类的相似图书候选"""
        if book_id not in self.book_features:
            return []
        feats = self.book_features[book_id]
        candidates = set()

        author = feats.get('author')
        if author in self.author_to_books:
            candidates.update(self.author_to_books[author][:topk])

        cat1 = feats.get('category1')
        if cat1 in self.category1_to_books:
            candidates.update(self.category1_to_books[cat1][:topk])

        cat2 = feats.get('category2')
        if cat2 in self.category2_to_books:
            candidates.update(self.category2_to_books[cat2][:topk])

        candidates.discard(book_id)
        return list(candidates)[:topk]

    def get_candidates(self, user_id, history_topk=50, popular_topk=200, similar_topk=10):
        """
        综合候选生成（增强版）：
        历史 + 全局热门 + 院系热门 + 院系协同 + 元数据相似 + 标题相似 + LightFM
        """
        history = self.user_history.get(user_id, [])
        history_books = [book_id for book_id, _, _ in history][-history_topk:]
        candidate_set = set(history_books)

        # 全局热门
        candidate_set.update(self._popular_books(topk=popular_topk))
        
        user_dept = self.user_info.get(user_id, {}).get('dept')
        if user_dept and user_dept in self.dept_popular_books:
            candidate_set.update(self.dept_popular_books[user_dept][:100])
        
        if user_dept and user_dept in self.dept_similarity:
            for similar_dept, similarity in self.dept_similarity[user_dept][:3]:  # top-3相似院系
                if similar_dept in self.dept_popular_books:
                    # 根据相似度选择数量
                    n_books = int(50 * similarity)
                    candidate_set.update(self.dept_popular_books[similar_dept][:n_books])

        # 来自用户备选的作者/分类相似
        for book_id in history_books:
            candidate_set.update(self._similar_by_meta(book_id, topk=similar_topk))
            candidate_set.update(self._title_similar_books(book_id, topk=max(3, similar_topk // 2)))

        if len(candidate_set) < popular_topk:
            candidate_set.update(self._popular_books(topk=popular_topk * 2))

        if self.lightfm_enabled:
            candidate_set.update(
                self._lightfm_top_items(user_id, topk=max(LIGHTFM_TOPK, similar_topk * 3))
            )

        return list(candidate_set)
    
    
    def visualize_statistics(self, output_dir='output/visualizations'):
        """在当前无图形环境下跳过可视化生成。"""
        print("📊 当前环境不支持可视化，已跳过图表生成。")

    def _plot_feature_importance(self, output_dir):
        """占位函数：可视化已禁用。"""
        return

def main():
    """
    主函数：执行完整的推荐流程
    
    步骤：
    1. 初始化推荐系统
    2. 加载数据
    3. 构建特征模型
    4. 生成可视化图表
    5. 为所有测试用户生成推荐
    6. 保存结果
    """
    print("=" * 80)
    print("📖 图书馆图书推荐系统 - 混合推荐算法")
    print("=" * 80)
    print()
    
    # 初始化推荐系统
    recommender = HybridRecommender()
    
    # 加载数据
    interactions = recommender.load_data()
    
    recommender.build_features(interactions)
    
    # 加载 LightFM 模型（如可用）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lightfm_model_path = os.path.join(project_root, 'models', 'lightfm_model.npz')
    recommender.load_lightfm_model(model_path=lightfm_model_path)

    # 生成可视化图表（当前环境跳过）
    # recommender.visualize_statistics()
    
    # 读取测试用户（从交互数据中提取所有用户）
    print("🎯 生成推荐结果...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    test = pd.read_csv(os.path.join(data_dir, 'inter_final_选手可见.csv'))
    test_users = test['user_id'].unique()
    print(f"   总用户数: {len(test_users)}")
    
    recommendations = []
    for idx, user_id in enumerate(test_users, 1):
        pred_book = recommender.recommend(user_id)
        recommendations.append({'user_id': user_id, 'book_id': pred_book})
        
        # 每100个用户显示一次进度
        if idx % 100 == 0:
            print(f"   处理进度: {idx}/{len(test_users)} ({idx/len(test_users)*100:.1f}%)")
    
    result_df = pd.DataFrame(recommendations)
    
    os.makedirs('output', exist_ok=True)
    output_file = 'output/submission_人民当家作组.csv'
    result_df.to_csv(output_file, index=False)
    
    print()
    print("=" * 80)
    print("✅ 推荐完成！")
    print("=" * 80)
    print(f"📊 生成推荐: {len(result_df)} 个用户")
    print(f"💾 结果文件: {output_file}")
    print("📈 可视化图表: 已跳过")
    print()
    
    # 输出一些统计信息
    print("📈 推荐统计:")
    print(f"   ✓ 唯一推荐书籍数: {result_df['book_id'].nunique()}")
    print(f"   ✓ 最常推荐的书籍: {result_df['book_id'].mode()[0] if len(result_df) > 0 else 'N/A'}")
    print(f"   ✓ 平均用户借阅次数: {np.mean(list(recommender.stats['user_borrow_counts'].values())):.1f}")
    print()

if __name__ == '__main__':
    main()
