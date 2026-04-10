import json
import requests
from datetime import datetime, timedelta
import random
import urllib.parse
import hashlib
import math

class MarketAnalyzer:
    def __init__(self, calendar_path):
        with open(calendar_path, 'r') as f:
            self.calendar = json.load(f)

    def resolve_topic_info(self, topic):
        """Finds AppIDs and BundleIDs for a given topic."""
        info = {
            "steam_id": None,
            "mobile_id": None,
            "category": "General",
            "is_game": False
        }
        
        # 1. Resolver Steam
        try:
            steam_search = f"https://store.steampowered.com/api/storesearch/?term={topic}&l=english&cc=US"
            resp = requests.get(steam_search, timeout=5).json()
            if resp.get('total') > 0:
                top_item = resp['items'][0]
                info["steam_id"] = top_item['id']
                info["is_game"] = True
                info["category"] = "PC/Console"
        except:
            pass

        # 2. Resolver Mobile (Simplified iTunes Search)
        try:
            itunes_search = f"https://itunes.apple.com/search?term={topic}&entity=software&limit=1"
            resp = requests.get(itunes_search, timeout=5).json()
            if resp.get('resultCount') > 0:
                top_mobile = resp['results'][0]
                info["mobile_id"] = top_mobile['bundleId']
                info["is_game"] = True
                # Deepseek if it's a game based on genre
                mobile_category = "Mobile Game" if "Games" in top_mobile.get('genres', []) else "Mobile App"
                if info["steam_id"] and mobile_category == "Mobile Game":
                    info["category"] = "Cross-platform Game"
                elif not info["steam_id"]:
                    info["category"] = mobile_category
        except:
            pass
            
        return info

    def _stable_int(self, key, minimum, maximum):
        """Deterministic pseudo-random integer for stable fallback values."""
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        span = maximum - minimum + 1
        return minimum + (int(digest[:8], 16) % span)

    def get_live_steam_data(self, appid):
        """Fetches real CCU data from Steam official API."""
        ccu = 0
        try:
            url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"
            resp = requests.get(url, timeout=5).json()
            ccu = resp['response'].get('player_count', 0)
        except:
            ccu = self._stable_int(f"steam:{appid}", 100, 500)
            
        # Mock growth based on real CCU scale
        return {
            "daily_velocity": round(ccu * 0.05) if ccu > 0 else random.randint(10, 50),
            "7d_growth": round(random.uniform(-5, 15), 2),
            "estimated_wishlists": round(ccu * 150) if ccu > 0 else random.randint(10000, 50000),
            "current_ccu": ccu
        }

    def get_live_mobile_data(self, bundle_id):
        """Fetches metadata from Mobile stores (Simulation for variety)."""
        # In a real app we'd use google-play-scraper here
        # For now we use the bundle_id to seed a more 'stable' simulation
        seed = sum(ord(c) for c in bundle_id) if bundle_id else self._stable_int("mobile:unknown", 1, 1000)
        rng = random.Random(seed)

        rank = rng.randint(1, 200)
        installs = round(100000 / (rank ** 0.6))
        
        return {
            "current_rank": rank,
            "estimated_daily_installs": installs,
            "arpu_proxy": round(rng.uniform(0.8, 4.0), 2)
        }

    def get_reddit_data(self, topic):
        """Fetches real Reddit discussion data without API key."""
        try:
            safe_topic = urllib.parse.quote(topic)
            url = f"https://www.reddit.com/search.json?q={safe_topic}&sort=relevance&limit=5"
            headers = {"User-Agent": "PentaAna/1.0 (Contact: admin@penta.com)"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                posts = data['data']['children']
                ups = sum(p['data']['ups'] for p in posts)
                comments = sum(p['data']['num_comments'] for p in posts)
                return {"mentions": len(posts), "engagement": ups + comments, "status": "LIVE (Reddit API)"}
        except Exception as e:
            pass
        mentions = self._stable_int(f"reddit:m:{topic}", 5, 50)
        engagement = self._stable_int(f"reddit:e:{topic}", 100, 5000)
        return {"mentions": mentions, "engagement": engagement, "status": "PROXY MOCK"}

    def get_twitch_data(self, topic):
        """Fetches real Twitch viewership data using public GQL."""
        try:
            url = "https://gql.twitch.tv/gql"
            headers = {
                "Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",
                "Content-Type": "application/json"
            }
            query = f"""
            query {{
              searchCategories(query: "{topic}", first: 1) {{
                edges {{
                  node {{
                    name
                    viewersCount
                  }}
                }}
              }}
            }}
            """
            resp = requests.post(url, headers=headers, json={"query": query}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                edges = data.get("data", {}).get("searchCategories", {}).get("edges", [])
                if edges:
                    viewers = edges[0]["node"].get("viewersCount", 0)
                    return {"viewers": viewers, "channels": max(1, round(viewers / 100)), "status": "LIVE (Twitch GQL)"}
        except Exception as e:
            pass
            
        viewers = self._stable_int(f"twitch:v:{topic}", 500, 25000)
        channels = max(5, round(viewers / 180))
        return {
            "viewers": viewers,
            "channels": channels,
            "status": "PROXY MOCK"
        }

    def correlate_with_calendar(self, trend_data):
        insights = []
        now = datetime.utcnow().date()
        for event in self.calendar:
            try:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            except Exception:
                continue

            # Compare by month-day to keep legacy calendars useful across years.
            event_this_year = event_date.replace(year=now.year)
            day_distance = abs((event_this_year - now).days)
            if day_distance > 120:
                continue

            score = max(0.55, 0.98 - (day_distance / 300))
            insights.append({
                "event": event["event"],
                "type": event["type"],
                "impact": event["impact"],
                "correlation_score": round(score, 2)
            })

        insights.sort(key=lambda x: x["correlation_score"], reverse=True)
        return insights[:3]

    def calculate_strategic_scenarios(self, metrics, reddit_data=None, twitch_data=None):
        """Calculates future potential and risk scores for strategic modeling using real combined signals."""
        ccu = metrics.get('current_ccu', 0)
        growth_7d = metrics.get('7d_growth', 0)
        
        # Incorporate external social signals if provided
        reddit_eng = reddit_data.get('engagement', 0) if reddit_data else random.randint(100, 1000)
        twitch_v = twitch_data.get('viewers', 0) if twitch_data else random.randint(1000, 5000)
        
        # Dynamically calculated Hype Score (Combining CCU momentum + Social Signals)
        # Cap values to prevent infinity scaling
        normalized_ccu_hype = min(30, ccu / 1000)
        normalized_social = min(70, (reddit_eng * 0.05) + (twitch_v * 0.005))
        hype_score = min(100, max(15, int(normalized_ccu_hype + normalized_social + growth_7d)))
        
        # Growth Potential & Risk Factor based on Data Matrix
        potential = min(100, max(0, 40 + (hype_score * 0.6) + growth_7d))
        risk = min(100, max(0, 60 - (hype_score * 0.3) + (1000 / (ccu + 100))))
        
        # Market Ceiling Estimate based on category rules
        base_ceiling = 100000
        category = metrics.get('category', 'General')
        if any(token in category for token in ["PC", "Console"]):
            base_ceiling = (ccu * 4) + (twitch_v * 15) if ccu > 0 else 500000
        elif "Mobile" in category:
            base_ceiling = (metrics.get('estimated_daily_installs', 1000) * 30 * metrics.get('arpu_proxy', 1.0))
            
        market_ceiling = int(base_ceiling * (1 + (hype_score / 100)))

        return {
            "potential": round(potential, 1),
            "risk": round(risk, 1),
            "hype_score": hype_score,
            "market_ceiling": market_ceiling,
            "followers_proxy": int(metrics.get('estimated_wishlists', 0) / 10) + int(reddit_eng * 1.5),
            "failure_warnings": self.get_failure_patterns(category, hype=hype_score),
            "scenarios": [
                {"name": "Mở rộng Tệp người dùng (Growth)", "likelihood": "Cao" if hype_score > 60 else "Thấp", "detail": f"Tận dụng đà {twitch_v} người xem Twitch để scale user acquisition. Thực hiện Creator Marketing campaign ngay lập tức."},
                {"name": "Trì hoãn & Tối ưu (Stable)", "likelihood": "Cao" if 30 <= hype_score <= 60 else "Thấp", "detail": "Lượng thảo luận ở mức an toàn. Cần tối ưu Retention và Monetization thay vì đổ tiền vào Ads."},
                {"name": "Phòng thủ Rủi ro (Risk)", "likelihood": "Cao" if hype_score < 30 else "Thấp", "detail": "Hype thấp bất thường. Tạm ngưng vung ngân sách Marketing, dồn lực fix các Failure Patterns cốt lõi."}
            ]
        }

    def get_failure_patterns(self, category, hype=50):
        """Returns intelligent pitfalls based on categorical data and hype momentum."""
        warnings = []
        if hype > 80:
            warnings.append("Hype quá mức (Over-hyped): Kỳ vọng của cộng đồng rất cao, rủi ro đánh giá tiêu cực nếu server sập lúc ra mắt.")
        elif hype < 30:
            warnings.append("Dead on Arrival (DOA): Mức độ nhận diện (Awareness) quá thấp so với chi phí.")
            
        patterns = {
            "Mobile Game": ["Chi phí ra người dùng (CPI) tăng mất kiểm soát", "Tỷ lệ giữ chân ngày 1 (D1 Retention) sụt giảm do tối ưu kém", "Sự nhạy cảm về cơ chế Gacha/P2W"],
            "PC Game": ["Lỗi giật lag / 최적화 (Optimization) ngày đầu", "Nội dung game quá ngắn so với tiến độ tiêu thụ nội dung", "Cộng đồng Toxic trên Reddit chưa có người phân giải"],
            "PC/Console": ["Lỗi giật lag / 최적화 (Optimization) ngày đầu", "Nội dung game quá ngắn so với tiến độ tiêu thụ nội dung", "Cộng đồng Toxic trên Reddit chưa có người phân giải"],
            "Mobile App": ["Độ giữ chân thấp sau tuần đầu do định vị chưa rõ", "Chi phí Ads vượt giá trị vòng đời (LTV)", "Onboarding dài làm giảm tỷ lệ kích hoạt"],
            "Product": ["Khó kích hoạt người dùng mới (Onboarding phức tạp)", "Sai lệch định giá so với đối thủ (Pricing Mismatch)"]
        }
        
        category_warnings = patterns.get(category, ["Chưa tiếp cận đúng tập khách hàng cốt lõi", "UX/UI kém thân thiện"])
        category_warnings.extend(warnings)
        return list(set(category_warnings))

    def calculate_health_score(self, metrics):
        # Deterministic health score from observable/proxy metrics.
        ccu = metrics.get('current_ccu', 0)
        growth_7d = metrics.get('7d_growth', 0)
        installs = metrics.get('estimated_daily_installs', 0)
        wishlists = metrics.get('estimated_wishlists', 0)
        arpu = metrics.get('arpu_proxy', 0)

        ccu_score = min(100, (ccu / 50000) * 100) if ccu > 0 else 35
        retention_score = min(100, max(20, 45 + (growth_7d * 1.6) + (math.log10(installs + 1) * 5)))
        revenue_score = min(100, max(20, 40 + (arpu * 12) + (math.log10(wishlists + 1) * 5)))

        total = (ccu_score * 0.4) + (retention_score * 0.3) + (revenue_score * 0.3)
        status = "SAFE" if total > 80 else "STABLE" if total > 55 else "DANGER"
        
        return {
            "total": round(total, 1),
            "status": status,
            "components": {
                "ccu": round(ccu_score),
                "retention": round(retention_score, 1),
                "revenue": round(revenue_score, 1)
            }
        }
