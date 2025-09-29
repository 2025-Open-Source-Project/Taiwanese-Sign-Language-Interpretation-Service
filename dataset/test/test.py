import json
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

class FrontendSignLanguageTester:
    def __init__(self, frontend_url="http://140.123.105.233:3000", videos_json_url=None):
        """
        初始化前端測試器
        
        Args:
            frontend_url (str): 前端HTML頁面的URL
            videos_json_url (str): videos.json的URL（可選）
        """
        self.frontend_url = frontend_url
        self.videos_json_url = videos_json_url or f"{frontend_url}/videos.json"
        self.driver = None
        self.test_data = None
        self.results = []
        self.video_db = set()
        
    def setup_driver(self, headless=True):
        """
        設置 Chrome WebDriver（使用 webdriver-manager 自動管理）
        """
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            print("Chrome WebDriver 初始化成功")
        except Exception as e:
            print(f"WebDriver 初始化失敗: {e}")
            print("請確認已安裝 ChromeDriver")
            raise
    
    def load_video_database(self):
        """
        載入影片資料庫 JSON
        """
        try:
            response = requests.get(self.videos_json_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.video_db = set(data.get('videos', []))
                print(f"成功載入影片資料庫，共 {len(self.video_db)} 個影片")
            else:
                print(f"載入影片資料庫失敗: HTTP {response.status_code}")
        except Exception as e:
            print(f"載入影片資料庫失敗: {e}")
            print("將使用空的影片資料庫進行測試")
    
    def load_test_data(self, file_path):
        """
        載入測試資料
        """
        try:
            # 嘗試多種編碼方式載入JSON
            for encoding in ['utf-8', 'utf-8-sig', 'big5', 'gbk']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        data = json.load(f)
                    print(f"成功使用 {encoding} 編碼載入測試資料")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            else:
                raise ValueError("無法使用任何編碼載入JSON檔案")
                
            self.test_data = pd.DataFrame(data)
            
            # 檢查必要欄位
            required_columns = ['text', 'can_find']
            missing_columns = [col for col in required_columns if col not in self.test_data.columns]
            
            if missing_columns:
                raise ValueError(f"測試資料缺少必要欄位: {missing_columns}")
                
            print(f"成功載入 {len(self.test_data)} 筆測試資料")
            
        except Exception as e:
            print(f"載入測試資料失敗: {e}")
            raise
    
    def wait_for_result(self, timeout=50): # 60
        """
        等待前端處理完成並回傳結果
        """
        try:
            # 等待結果出現
            result_element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.ID, "result"))
            )
            
            # 等待結果不再是 "處理中，請稍候..."
            WebDriverWait(self.driver, timeout).until(
                lambda driver: "處理中" not in driver.find_element(By.ID, "result").text
            )
            
            return result_element.text
            
        except TimeoutException:
            print("等待結果超時")
            return None
    
    def submit_text_to_frontend(self, text):
        """
        將文字提交到前端並取得結果
        """
        try:
            # 導航到前端頁面
            self.driver.get(self.frontend_url)
            
            # 等待頁面載入
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "textInput"))
            )
            
            # 清空並輸入文字
            text_input = self.driver.find_element(By.ID, "textInput")
            text_input.clear()
            text_input.send_keys(text)
            
            # 點擊送出按鈕
            submit_button = self.driver.find_element(By.XPATH, "//button[contains(text(), '送出查詢')]")
            submit_button.click()
            
            # 等待並取得結果
            result_text = self.wait_for_result()
            
            return result_text
            
        except Exception as e:
            print(f"提交文字到前端失敗: {e}")
            return None
    
    def parse_frontend_result(self, result_text):
        """
        解析前端回傳的結果
        """
        if not result_text:
            return []
            
        results = []
        lines = result_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if '→' in line:
                # 解析格式：句子 → 路徑 或 句子 → 沒有相似的句子
                parts = line.split('→')
                if len(parts) == 2:
                    sentence = parts[0].strip()
                    animation_path = parts[1].strip()
                    results.append({
                        'sentence': sentence,
                        'animation_path': animation_path
                    })
        
        return results
    
    def evaluate_frontend_result(self, parsed_results, expected_can_find, expected_animation=None):
        """
        評估前端結果（考慮影片資料庫檢查）
        """
        if not parsed_results:
            return 0, False, "無法解析前端結果"
        
        all_found = True
        details = []
        
        for result in parsed_results:
            animation_path = result.get('animation_path', '')
            sentence = result.get('sentence', '')
            
            if animation_path == '沒有相似的句子':
                all_found = False
                details.append(f"{sentence} -> 沒有相似的句子")
            else:
                # 檢查影片是否在資料庫中（模擬前端邏輯）
                if self.video_db and animation_path not in self.video_db:
                    # 如果影片不在資料庫中，前端會顯示為沒有相似的句子
                    all_found = False
                    details.append(f"{sentence} -> 沒有相似的句子 (影片路徑不存在)")
                else:
                    details.append(f"{sentence} -> {animation_path}")
        
        predicted_can_find = 1 if all_found else 0
        is_correct = (predicted_can_find == expected_can_find)
        
        # 額外檢查：如果預期找得到，檢查動畫路徑是否正確
        if expected_can_find == 1 and predicted_can_find == 1 and expected_animation:
            found_expected_animation = any(
                expected_animation in result.get('animation_path', '') 
                for result in parsed_results
            )
            if not found_expected_animation:
                is_correct = False
                details.append(f"預期動畫路徑: {expected_animation}")
        
        return predicted_can_find, is_correct, "; ".join(details)
    
    def run_tests(self):
        """
        執行所有測試
        """
        if not self.driver:
            raise ValueError("請先設置 WebDriver")
        if self.test_data is None:
            raise ValueError("請先載入測試資料")
        
        print(f"開始執行 {len(self.test_data)} 個前端測試...")
        print("=" * 60)
        
        self.results = []
        correct_count = 0
        total_count = len(self.test_data)
        
        for index, row in self.test_data.iterrows():
            text = str(row['text'])
            expected_can_find = int(row['can_find'])
            test_id = row.get('id', index)
            expected_animation = row.get('sign_animation', None)
            
            print(f"測試 {test_id}: {text[:50]}...")
            
            # 提交到前端
            result_text = self.submit_text_to_frontend(text)
            
            if result_text is None:
                print("  ❌ 前端測試失敗")
                result = {
                    'test_id': test_id,
                    'index': index,
                    'text': text,
                    'expected': expected_can_find,
                    'predicted': -1,
                    'is_correct': False,
                    'frontend_result': None,
                    'error': '前端測試失敗',
                    'details': '前端測試失敗',
                    'expected_animation': expected_animation
                }
            else:
                # 解析前端結果
                parsed_results = self.parse_frontend_result(result_text)
                predicted_can_find, is_correct, details = self.evaluate_frontend_result(
                    parsed_results, expected_can_find, expected_animation
                )
                
                if is_correct:
                    correct_count += 1
                    print(f"  ✅ 正確 (預期: {expected_can_find}, 實際: {predicted_can_find})")
                else:
                    print(f"  ❌ 錯誤 (預期: {expected_can_find}, 實際: {predicted_can_find})")
                    print(f"    詳細: {details}")
                
                result = {
                    'test_id': test_id,
                    'index': index,
                    'text': text,
                    'expected': expected_can_find,
                    'predicted': predicted_can_find,
                    'is_correct': is_correct,
                    'frontend_result': result_text,
                    'parsed_results': parsed_results,
                    'error': None,
                    'details': details,
                    'expected_animation': expected_animation
                }
            
            self.results.append(result)
            
            # 避免過快提交，給前端一些處理時間
            time.sleep(2)
        
        accuracy = correct_count / total_count
        print("=" * 60)
        print(f"前端測試完成！準確率: {accuracy:.2%} ({correct_count}/{total_count})")
        
        return self.results
    
    def generate_confusion_matrix(self):
        """
        生成混淆矩陣
        """
        if not self.results:
            raise ValueError("請先執行測試")
            
        # 過濾掉前端失敗的結果
        valid_results = [r for r in self.results if r['predicted'] != -1]
        
        if not valid_results:
            print("沒有有效的測試結果")
            return
            
        y_true = [r['expected'] for r in valid_results]
        y_pred = [r['predicted'] for r in valid_results]
        
        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        
        # 繪製混淆矩陣
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cannot Find (0)', 'Can Find (1)'], 
                   yticklabels=['Cannot Find (0)', 'Can Find (1)'])
        plt.title('Semantic Determination Confusion Matrix')
        plt.xlabel('Predicted Result')
        plt.ylabel('Actual Result')
        plt.tight_layout()
        
        # 儲存混淆矩陣圖片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'frontend_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已儲存為 frontend_confusion_matrix_{timestamp}.png")
        
        plt.show()
        
        # 印出分類報告
        print("\n分類報告：")
        print("=" * 50)
        report = classification_report(y_true, y_pred, 
                                     target_names=['找不到', '找得到'], 
                                     digits=4)
        print(report)
        
        return cm
    
    def save_error_cases(self, output_file=None):
        """
        儲存錯誤案例到檔案
        """
        if not self.results:
            raise ValueError("請先執行測試")
            
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"frontend_error_cases_{timestamp}.txt"
            
        error_cases = [r for r in self.results if not r['is_correct']]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("前端語意判斷系統錯誤案例分析\n")
            f.write("=" * 60 + "\n")
            f.write(f"總測試數量: {len(self.results)}\n")
            f.write(f"錯誤案例數量: {len(error_cases)}\n")
            f.write(f"錯誤率: {len(error_cases)/len(self.results):.2%}\n\n")
            
            # 按錯誤類型分類
            frontend_failures = [case for case in error_cases if case['predicted'] == -1]
            false_positives = [case for case in error_cases if case['expected'] == 0 and case['predicted'] == 1]
            false_negatives = [case for case in error_cases if case['expected'] == 1 and case['predicted'] == 0]
            
            f.write(f"前端失敗案例: {len(frontend_failures)}\n")
            f.write(f"誤判為找得到 (False Positive): {len(false_positives)}\n")
            f.write(f"誤判為找不到 (False Negative): {len(false_negatives)}\n\n")
            
            for i, case in enumerate(error_cases, 1):
                f.write(f"錯誤案例 {i} (ID: {case.get('test_id', 'N/A')}):\n")
                f.write(f"輸入文字: {case['text']}\n")
                f.write(f"預期結果: {case['expected']} ({'找得到' if case['expected'] == 1 else '找不到'})\n")
                f.write(f"實際結果: {case['predicted']} ({'找得到' if case['predicted'] == 1 else '找不到' if case['predicted'] == 0 else '前端失敗'})\n")
                
                if case.get('expected_animation'):
                    f.write(f"預期動畫路徑: {case['expected_animation']}\n")
                    
                if case.get('details'):
                    f.write(f"詳細結果: {case['details']}\n")
                
                if case.get('frontend_result'):
                    f.write(f"前端原始回應:\n{case['frontend_result']}\n")
                elif case.get('error'):
                    f.write(f"錯誤原因: {case['error']}\n")
                    
                f.write("-" * 50 + "\n\n")
                
        print(f"錯誤案例已儲存到 {output_file}")
        return output_file
    
    def close(self):
        """
        關閉 WebDriver
        """
        if self.driver:
            self.driver.quit()
            print("WebDriver 已關閉")


def main():
    """
    主函數
    """
    # 設定參數
    frontend_url = "http://140.123.105.233:3000/"  # 修改為的前端URL
    test_file = "test_datafile.json"
    
    # 建立測試器
    tester = FrontendSignLanguageTester(frontend_url=frontend_url)
    
    try:
        print("設置 WebDriver...")
        # 設置 WebDriver（headless=False 可以看到瀏覽器操作）
        tester.setup_driver(headless=True)
        
        print("載入影片資料庫...")
        # 載入影片資料庫
        tester.load_video_database()
        
        print("載入測試資料...")
        # 載入測試資料
        tester.load_test_data(test_file)
        
        print("開始執行前端測試...")
        # 執行測試
        results = tester.run_tests()
        
        print("生成混淆矩陣...")
        # 生成混淆矩陣
        tester.generate_confusion_matrix()
        
        print("儲存錯誤案例...")
        # 儲存錯誤案例
        tester.save_error_cases()
        
        print("前端測試完成！")
        
    except Exception as e:
        print(f"測試執行失敗: {e}")
        print("請確認：")
        print("1. 前端服務是否正常運行")
        print("2. ChromeDriver 是否已安裝")
        print("3. 測試檔案是否存在")
        print("4. 前端URL是否正確")
        
    finally:
        # 關閉 WebDriver
        tester.close()


if __name__ == "__main__":
    main()