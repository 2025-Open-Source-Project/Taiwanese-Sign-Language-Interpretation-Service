### How does the test data set work
1. can_find -> 是否應該可以從資料庫中找到 (0: 不行 ; 1: 可以) 
2. text -> 輸入之查詢句
3. result -> 預期得到的資料庫相似句
4. sign_animation -> 預期得到的動畫路徑

**當 can_find = 0 時，輸出應為 「沒有相似的句子」**
**判斷對錯依據為 can_find && result && sign_animation。**
### 例如:
![err msg][螢幕擷取畫面 2025-09-17 004106.png]  

### Confusion Matrix
![][2nd\frontend_confusion_matrix_20250923_212534.png]  

### How to run

:::danger Precaution
1. need connect to CCU VPN
2. need python version >= 3.7
:::
```python test.py```