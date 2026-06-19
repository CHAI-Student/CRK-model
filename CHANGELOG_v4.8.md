# v4.8 변경사항 - stock_qty/product_weight 활용 개선 + has_loadcell 필드 지원

**날짜**: 2026-02-04
**버전**: v4.8

## 변경 개요

1. **stock_qty 활용 개선**: active_products 사용 시 실제 재고 값을 count_calculator에 전달
2. **product_weight 우선순위 개선**: loadcell-only 폴백 시 Node.js 최신 무게 우선 사용
3. **has_loadcell 필드 지원**: 로드셀 없는 상품 Vision-only 모드 처리

## 수정된 파일

### 1. count_calculator.py

**위치**: `services/model/src/weight/count_calculator.py`

#### 변경 1-1: 라인 136-143 - 실제 stock 값 사용

**기존 코드**:
```python
# v4.7: active_products는 이미 stock 필터링 완료 (stock > 0)
# 따라서 재고 필터링 스킵

# 개수 추정 (v4.7: stock 상한 적용하지 않음, 이미 필터링됨)
count = self._estimate_count(abs_weight, product_weight, stock=0)
if count <= 0:
    continue
```

**수정 후**:
```python
# v4.8: active_products의 실제 stock_qty를 사용하여 상한 적용
# stock=0이면 재고 소진 상태이므로 count=0 처리
count = self._estimate_count(abs_weight, product_weight, stock=stock)
if count <= 0:
    logger.info(
        f"[COUNT] v4.8: Stock depleted: {product_name} "
        f"(stock={stock}, vision detected but filtered)"
    )
    continue
```

**이유**: ActiveProductStore는 매핑만 하고 stock=0인 상품도 포함 가능. 실제 재고 상한을 정확히 적용해야 count 계산이 올바름.

#### 변경 1-2: 라인 200-230 - has_loadcell Vision-only 로직 추가

**기존 코드**:
```python
if product.weight <= 0:
    logger.debug(f"Skipping product with zero weight: {product.name}")
    continue

# 개수 추정 (v4.3: 재고 상한 적용)
count = self._estimate_count(abs_weight, product.weight, stock=product.stock)
if count <= 0:
    continue

# 예상 무게 계산
expected_weight = product.weight * count
weight_error = abs(abs_weight - expected_weight)

# 허용 오차: 고정 5g 사용
tolerance_amount = self.tolerance_grams

# 검증
validated = weight_error <= tolerance_amount

# 매칭 점수 계산
match_score = self._calculate_match_score(
    weight_error=weight_error,
    expected_weight=expected_weight,
    vision_confidence=candidate.combined_confidence,
)
```

**수정 후**:
```python
if product.weight <= 0:
    logger.debug(f"Skipping product with zero weight: {product.name}")
    continue

# v4.8: has_loadcell 필드 지원 (Vision-only 모드)
has_loadcell = getattr(product, 'has_loadcell', 'true')  # 기본값: true
if has_loadcell in ['false', 'null']:
    # 로드셀 없음 → 무게 검증 스킵, Vision 신뢰도만 사용
    logger.info(
        f"[COUNT] v4.8: Vision-only mode: {product_name} (has_loadcell={has_loadcell})"
    )
    count = 1  # Vision이 감지했으면 1개로 간주
    validated = True  # 무게 검증 스킵
    match_score = candidate.combined_confidence  # Vision 신뢰도만 사용
    expected_weight = 0.0
    weight_error = 0.0
else:
    # 로드셀 있음 → 기존 무게 검증 로직
    # 개수 추정 (v4.3: 재고 상한 적용)
    count = self._estimate_count(abs_weight, product.weight, stock=product.stock)
    if count <= 0:
        continue

    # 예상 무게 계산
    expected_weight = product.weight * count
    weight_error = abs(abs_weight - expected_weight)

    # 허용 오차: 고정 5g 사용
    tolerance_amount = self.tolerance_grams

    # 검증
    validated = weight_error <= tolerance_amount

    # 매칭 점수 계산
    match_score = self._calculate_match_score(
        weight_error=weight_error,
        expected_weight=expected_weight,
        vision_confidence=candidate.combined_confidence,
    )
```

**이유**: 로드셀 없는 상품은 무게 검증을 스킵하고 Vision 신뢰도만으로 판단.

---

### 2. decision_engine.py

**위치**: `services/model/src/engine/decision_engine.py`

#### 변경 2-1: 라인 139-141, 153-155 - active_products 전달

**기존 코드** (라인 139-141):
```python
# 1. 후보군이 없는 경우 → Loadcell-only 폴백
if not vision_candidates:
    logger.warning("No vision candidates provided, trying loadcell-only fallback")
    return self.judge_by_weight_only(delta_weight, timestamp)
```

**수정 후**:
```python
# 1. 후보군이 없는 경우 → Loadcell-only 폴백
if not vision_candidates:
    logger.warning("No vision candidates provided, trying loadcell-only fallback")
    return self.judge_by_weight_only(delta_weight, timestamp, active_products=active_products)
```

**기존 코드** (라인 153-155):
```python
if not estimates:
    logger.warning("No valid count estimates, trying loadcell-only fallback")
    return self.judge_by_weight_only(delta_weight, timestamp)
```

**수정 후**:
```python
if not estimates:
    logger.warning("No valid count estimates, trying loadcell-only fallback")
    return self.judge_by_weight_only(delta_weight, timestamp, active_products=active_products)
```

**이유**: loadcell-only 폴백 시 active_products의 최신 무게 정보를 사용하기 위해 전달.

#### 변경 2-2: 라인 321-435 - judge_by_weight_only() 시그니처 및 로직 수정

**기존 시그니처**:
```python
def judge_by_weight_only(
    self,
    delta_weight: float,
    timestamp: Optional[float] = None,
) -> JudgmentResult:
```

**수정 후**:
```python
def judge_by_weight_only(
    self,
    delta_weight: float,
    timestamp: Optional[float] = None,
    active_products: Optional[List] = None,  # v4.8: 추가
) -> JudgmentResult:
```

**기존 로직 시작부** (라인 339-352):
```python
abs_weight = abs(delta_weight)

logger.info(f"Loadcell-only fallback: delta_weight={delta_weight:.1f}g")

# 무게 변화가 너무 작은 경우
if abs_weight < self.min_weight_change:
    logger.info(f"Weight change too small for fallback: {abs_weight:.1f}g")
    return self._create_no_detection_result(delta_weight, timestamp)

# 모든 상품에서 가장 가까운 무게 찾기
all_products = self.product_db.get_all_products()
```

**수정 후**:
```python
abs_weight = abs(delta_weight)

logger.info(f"Loadcell-only fallback: delta_weight={delta_weight:.1f}g")

# 무게 변화가 너무 작은 경우
if abs_weight < self.min_weight_change:
    logger.info(f"Weight change too small for fallback: {abs_weight:.1f}g")
    return self._create_no_detection_result(delta_weight, timestamp)

# v4.8: active_products 우선 사용 (Node.js 최신 무게)
candidate_products = []
if active_products:
    for ap in active_products:
        # has_loadcell 확인: "false"/"null"이면 제외
        has_loadcell = getattr(ap, 'has_loadcell', 'true')
        if has_loadcell in ['false', 'null']:
            logger.debug(
                f"[LOADCELL-ONLY] v4.8: Skip no-loadcell product: {ap.product_name}"
            )
            continue

        if ap.yolo_class_id is not None and ap.product_weight > 0 and ap.stock_qty > 0:
            # ProductInfo를 ProductDB 형식으로 변환
            pseudo_product = type('PseudoProduct', (), {
                'product_id': ap.yolo_class_id,
                'product_idx': ap.product_idx,
                'name': ap.product_name,
                'weight': ap.product_weight,  # Node.js 최신 무게
                'price': ap.sale_price,
                'stock': ap.stock_qty,
                'has_loadcell': has_loadcell,
            })()
            candidate_products.append(pseudo_product)

    if candidate_products:
        logger.info(
            f"[LOADCELL-ONLY] v4.8: Using {len(candidate_products)} products "
            f"from active_products (Node.js latest weights)"
        )

# Fallback: ProductDatabase에서 조회
if not candidate_products:
    all_products = self.product_db.get_all_products()
    # has_loadcell 필터링
    candidate_products = [
        p for p in all_products
        if p.weight > 0 and getattr(p, 'has_loadcell', 'true') not in ['false', 'null']
    ]
    if not candidate_products:
        logger.warning("No loadcell-enabled products for fallback")
        return self._create_no_detection_result(delta_weight, timestamp)
    logger.info(
        f"[LOADCELL-ONLY] v4.8: Using {len(candidate_products)} products "
        f"from ProductDB (fallback)"
    )
```

**기존 반복문** (라인 354-362):
```python
if not all_products:
    logger.warning("No products in database for fallback")
    return self._create_no_detection_result(delta_weight, timestamp)

best_match = None
best_error = float('inf')
best_count = 0

for product in all_products:
```

**수정 후**:
```python
# 기존 로직: 가장 가까운 무게 찾기
best_match = None
best_error = float('inf')
best_count = 0

for product in candidate_products:
```

**이유**:
- active_products가 있으면 Node.js 최신 무게 우선 사용
- has_loadcell="false"/"null" 상품 제외
- ProductDB fallback도 has_loadcell 필터링 적용

---

### 3. multi_zone.py

**위치**: `services/model/src/api/routes/multi_zone.py`

#### 변경 3-1: 라인 68-76 - ProductInfo 필드명 변경

**기존 코드**:
```python
class ProductInfo(BaseModel):
    """Node.js에서 전달하는 상품 정보."""

    product_idx: str = Field(..., description="상품 ID (IF11)")
    product_name: str = Field(..., description="상품명")
    sale_price: int = Field(..., description="판매가격")
    product_weight: Optional[str] = Field(default="0", description="상품 무게 (g), 없으면 0")
    stock_qty: Optional[int] = Field(default=None, description="재고 수량 (v4.6: None이면 무제한)")
    loadcell: str = Field(default="false", description="로드셀 사용 여부")
```

**수정 후**:
```python
class ProductInfo(BaseModel):
    """Node.js에서 전달하는 상품 정보."""

    product_idx: str = Field(..., description="상품 ID (IF11)")
    product_name: str = Field(..., description="상품명")
    sale_price: int = Field(..., description="판매가격")
    product_weight: Optional[str] = Field(default="0", description="상품 무게 (g), 없으면 0")
    stock_qty: Optional[int] = Field(default=None, description="재고 수량 (v4.6: None이면 무제한)")
    has_loadcell: str = Field(default="true", description="로드셀 사용 여부 (v4.8)")
```

**이유**: Node.js가 전송하는 필드명과 일치 (loadcell → has_loadcell).

#### 변경 3-2: 라인 729-755 - ActiveProductStore 저장 시 has_loadcell 포함

**기존 코드**:
```python
# 상품 정보를 dict 리스트로 변환 (v4.6: stock_qty None 처리)
products_dict = []
for p in request.products:
    # stock_qty가 None이면 999 (무제한)로 처리 (v4.6)
    stock = p.stock_qty if p.stock_qty is not None else 999
    products_dict.append({
        "product_idx": p.product_idx,
        "product_name": p.product_name,
        "sale_price": p.sale_price,
        "product_weight": p.product_weight or "0",
        "stock_qty": stock,
    })
```

**수정 후**:
```python
# 상품 정보를 dict 리스트로 변환 (v4.6: stock_qty None 처리)
products_dict = []
for p in request.products:
    # stock_qty가 None이면 999 (무제한)로 처리 (v4.6)
    stock = p.stock_qty if p.stock_qty is not None else 999
    products_dict.append({
        "product_idx": p.product_idx,
        "product_name": p.product_name,
        "sale_price": p.sale_price,
        "product_weight": p.product_weight or "0",
        "stock_qty": stock,
        "has_loadcell": p.has_loadcell,  # v4.8: 추가
    })
```

---

### 4. product_db.py (models.py)

**위치**: `services/model/src/engine/models.py`

#### 변경 4-1: 라인 235-265 - ProductInfo 모델에 has_loadcell 필드 추가

**기존 코드**:
```python
@dataclass
class ProductInfo:
    """
    상품 정보 (데이터베이스용).

    Attributes:
        product_id: 상품 ID (내부 ID)
        name: 상품 이름 (한글)
        category: 카테고리
        weight: 단위 무게 (g)
        price: 가격 (원)
        barcode: 바코드 (선택)
        stock: 재고 수량 (선택)
        image_count: 등록된 이미지 수
        yolo_class_id: YOLO 모델의 클래스 ID (선택)
        yolo_class_name: YOLO 모델의 클래스 이름 (선택)
        product_idx: IF11 상품 ID (선택, 예: "P17689508539755305")
    """
    product_id: int
    name: str
    category: str
    weight: float  # grams
    price: int     # won
    barcode: Optional[str] = None
    stock: int = 0
    image_count: int = 0
    # YOLO-IF11 매핑 필드
    yolo_class_id: Optional[int] = None
    yolo_class_name: Optional[str] = None
    product_idx: Optional[str] = None
```

**수정 후**:
```python
@dataclass
class ProductInfo:
    """
    상품 정보 (데이터베이스용).

    Attributes:
        product_id: 상품 ID (내부 ID)
        name: 상품 이름 (한글)
        category: 카테고리
        weight: 단위 무게 (g)
        price: 가격 (원)
        barcode: 바코드 (선택)
        stock: 재고 수량 (선택)
        image_count: 등록된 이미지 수
        yolo_class_id: YOLO 모델의 클래스 ID (선택)
        yolo_class_name: YOLO 모델의 클래스 이름 (선택)
        product_idx: IF11 상품 ID (선택, 예: "P17689508539755305")
        has_loadcell: 로드셀 사용 여부 (v4.8, "true"/"false"/"null")
    """
    product_id: int
    name: str
    category: str
    weight: float  # grams
    price: int     # won
    barcode: Optional[str] = None
    stock: int = 0
    image_count: int = 0
    # YOLO-IF11 매핑 필드
    yolo_class_id: Optional[int] = None
    yolo_class_name: Optional[str] = None
    product_idx: Optional[str] = None
    has_loadcell: str = "true"  # v4.8: 추가 (기본값: true)
```

**위치**: `services/model/src/database/product_db.py`

#### 변경 4-2: 라인 366-424 - update_product() 메서드에 has_loadcell 파라미터 추가

**기존 시그니처** (라인 366-375):
```python
def update_product(
    self,
    product_id: int,
    name: Optional[str] = None,
    category: Optional[str] = None,
    weight: Optional[float] = None,
    price: Optional[int] = None,
    barcode: Optional[str] = None,
    stock: Optional[int] = None,
) -> bool:
```

**수정 후**:
```python
def update_product(
    self,
    product_id: int,
    name: Optional[str] = None,
    category: Optional[str] = None,
    weight: Optional[float] = None,
    price: Optional[int] = None,
    barcode: Optional[str] = None,
    stock: Optional[int] = None,
    has_loadcell: Optional[str] = None,  # v4.8: 추가
) -> bool:
```

**기존 로직** (라인 409-422):
```python
# 필드 업데이트 (dataclass는 frozen이 아니므로 직접 수정 가능)
if name is not None:
    product.name = name
if category is not None:
    product.category = category
if weight is not None:
    product.weight = float(weight)
if price is not None:
    product.price = int(price)
if barcode is not None:
    product.barcode = barcode
if stock is not None:
    product.stock = int(stock)

logger.info(f"Product updated: id={product_id}")
return True
```

**수정 후**:
```python
# 필드 업데이트 (dataclass는 frozen이 아니므로 직접 수정 가능)
if name is not None:
    product.name = name
if category is not None:
    product.category = category
if weight is not None:
    product.weight = float(weight)
if price is not None:
    product.price = int(price)
if barcode is not None:
    product.barcode = barcode
if stock is not None:
    product.stock = int(stock)
if has_loadcell is not None:  # v4.8: 추가
    product.has_loadcell = has_loadcell

logger.info(f"Product updated: id={product_id}")
return True
```

---

### 5. active_product_store.py

**위치**: `services/model/src/session/active_product_store.py`

#### 변경 5-1: 라인 42-50 - ProductInfo 데이터 클래스에 has_loadcell 추가

**기존 코드**:
```python
@dataclass
class ProductInfo:
    """Node.js에서 받은 상품 정보."""

    product_idx: str
    product_name: str
    sale_price: int
    product_weight: float
    stock_qty: int
    yolo_class_id: Optional[int] = None  # 매핑된 YOLO 클래스 ID
```

**수정 후**:
```python
@dataclass
class ProductInfo:
    """Node.js에서 받은 상품 정보."""

    product_idx: str
    product_name: str
    sale_price: int
    product_weight: float
    stock_qty: int
    yolo_class_id: Optional[int] = None  # 매핑된 YOLO 클래스 ID
    has_loadcell: str = "true"  # v4.8: 추가
```

#### 변경 5-2: 라인 210-238 - set_products() 메서드에 has_loadcell 처리 추가

**기존 코드**:
```python
for p in products:
    product_name = p.get("product_name", "")
    product_idx = p.get("product_idx", "")
    sale_price = int(p.get("sale_price", 0))
    stock_qty = int(p.get("stock_qty", 0))

    # product_weight는 문자열일 수 있음
    weight_str = p.get("product_weight", "0")
    try:
        product_weight = float(weight_str) if weight_str else 0.0
    except (ValueError, TypeError):
        product_weight = 0.0

    # YOLO class_id 찾기
    yolo_class_id = self._find_yolo_class_id(product_name)

    if yolo_class_id is None:
        unmapped_names.append(product_name)
        continue

    # ProductInfo 생성
    product_info = ProductInfo(
        product_idx=product_idx,
        product_name=product_name,
        sale_price=sale_price,
        product_weight=product_weight,
        stock_qty=stock_qty,
        yolo_class_id=yolo_class_id,
    )
```

**수정 후**:
```python
for p in products:
    product_name = p.get("product_name", "")
    product_idx = p.get("product_idx", "")
    sale_price = int(p.get("sale_price", 0))
    stock_qty = int(p.get("stock_qty", 0))
    has_loadcell = p.get("has_loadcell", "true")  # v4.8: 추가

    # product_weight는 문자열일 수 있음
    weight_str = p.get("product_weight", "0")
    try:
        product_weight = float(weight_str) if weight_str else 0.0
    except (ValueError, TypeError):
        product_weight = 0.0

    # YOLO class_id 찾기
    yolo_class_id = self._find_yolo_class_id(product_name)

    if yolo_class_id is None:
        unmapped_names.append(product_name)
        continue

    # ProductInfo 생성
    product_info = ProductInfo(
        product_idx=product_idx,
        product_name=product_name,
        sale_price=sale_price,
        product_weight=product_weight,
        stock_qty=stock_qty,
        yolo_class_id=yolo_class_id,
        has_loadcell=has_loadcell,  # v4.8: 추가
    )
```

---

### 6. judgment_service.py

**위치**: `services/model/src/service/judgment_service.py`

#### 변경 6-1: 라인 20-28 - ProductInfo 데이터 클래스에 has_loadcell 추가

**기존 코드**:
```python
@dataclass
class ProductInfo:
    """상품 정보 (Node.js 요청)."""
    product_idx: str
    product_name: str
    sale_price: int
    product_weight: str
    stock_qty: int = 0
```

**수정 후**:
```python
@dataclass
class ProductInfo:
    """상품 정보 (Node.js 요청)."""
    product_idx: str
    product_name: str
    sale_price: int
    product_weight: str
    stock_qty: int = 0
    has_loadcell: str = "true"  # v4.8: 추가
```

#### 변경 6-2: 라인 166-187 - _sync_product_database()에 has_loadcell 동기화 추가

**기존 코드**:
```python
def _sync_product_database(self, products: List[ProductInfo]) -> None:
    """[추가] Node.js로부터 받은 실시간 상품 정보를 DB에 동기화."""
    for p in products:
        class_id = self._product_db.get_yolo_class_id_by_product_idx(p.product_idx)
        if class_id is not None:
            try:
                # 1. 무게 업데이트
                weight = float(p.product_weight) if p.product_weight else 0.0
                if weight > 0:
                    self._product_db.update_weight(class_id, weight)

                # 2. 재고 및 가격 업데이트
                # update_product를 통해 DB 내의 stock 값을 0보다 크게 만들어 필터링 방지
                self._product_db.update_product(
                    product_id=class_id,
                    price=p.sale_price,
                    stock=p.stock_qty,
                    weight=weight if weight > 0 else None
                )
                logger.debug(f"[SYNC] Product {p.product_idx} (ID:{class_id}) updated: weight={weight}g, stock={p.stock_qty}")
            except (ValueError, TypeError) as e:
                logger.warning(f"[SYNC] Failed to update product {p.product_idx}: {e}")
```

**수정 후**:
```python
def _sync_product_database(self, products: List[ProductInfo]) -> None:
    """Node.js로부터 받은 실시간 상품 정보를 DB에 동기화 (v4.8: has_loadcell 추가)."""
    for p in products:
        class_id = self._product_db.get_yolo_class_id_by_product_idx(p.product_idx)
        if class_id is not None:
            try:
                # 1. 무게 업데이트
                weight = float(p.product_weight) if p.product_weight else 0.0
                if weight > 0:
                    self._product_db.update_weight(class_id, weight)

                # 2. 재고/가격/has_loadcell 업데이트 (v4.8)
                self._product_db.update_product(
                    product_id=class_id,
                    price=p.sale_price,
                    stock=p.stock_qty,
                    weight=weight if weight > 0 else None,
                    has_loadcell=p.has_loadcell,  # v4.8: 추가
                )
                logger.debug(
                    f"[SYNC] Product {p.product_idx} (ID:{class_id}) updated: "
                    f"weight={weight}g, stock={p.stock_qty}, has_loadcell={p.has_loadcell}"
                )
            except (ValueError, TypeError) as e:
                logger.warning(f"[SYNC] Failed to update product {p.product_idx}: {e}")
```

---

## 테스트 시나리오

### 시나리오 1: Vision 성공 + stock_qty 활용

**요청**:
```json
{
  "products": [
    {
      "product_idx": "26",
      "product_name": "치킨마요주먹밥",
      "sale_price": 3500,
      "product_weight": "367",
      "stock_qty": 5,
      "has_loadcell": "true"
    }
  ]
}
```

**기대 결과**:
- ✅ count_calculator가 active_products 무게(367g) 사용
- ✅ stock 상한(5개) 정확히 적용
- ✅ count=1 (not 0)
- ✅ 로그: `[COUNT] v4.8: Using active_product: 치킨마요...`

### 시나리오 2: Vision 실패 + loadcell-only 폴백

**요청**: 동일, Vision 감지 실패

**기대 결과**:
- ✅ judge_by_weight_only() 호출
- ✅ active_products의 최신 무게(367g) 사용
- ✅ ProductDB 무게(365g) 사용 안 함
- ✅ 치킨마요 매칭 성공
- ✅ 로그: `[LOADCELL-ONLY] v4.8: Using ... from active_products`

### 시나리오 3: stock=0 필터링

**요청**:
```json
{
  "products": [
    {
      "product_idx": "26",
      "product_name": "치킨마요주먹밥",
      "sale_price": 3500,
      "product_weight": "367",
      "stock_qty": 0,
      "has_loadcell": "true"
    }
  ]
}
```

**기대 결과**:
- ✅ count_calculator가 stock=0 감지
- ✅ count=0으로 필터링
- ✅ 로그: `[COUNT] v4.8: Stock depleted: 치킨마요...`
- ✅ 최종: NO_DETECTION 또는 다른 후보군으로 진행

### 시나리오 4: has_loadcell="false" 처리 (Vision-only)

**요청**:
```json
{
  "products": [
    {
      "product_idx": "100",
      "product_name": "콜라",
      "sale_price": 1500,
      "product_weight": "0",
      "stock_qty": 10,
      "has_loadcell": "false"
    }
  ]
}
```

**Vision**: 콜라 감지 (신뢰도 0.92)
**Delta**: -250g

**기대 결과**:
- ✅ 무게 검증 스킵
- ✅ count=1 (Vision 감지했으므로)
- ✅ 로그: `[COUNT] v4.8: Vision-only mode: 콜라...`
- ✅ match_score = vision_confidence (0.92)

### 시나리오 5: has_loadcell="false" + loadcell-only 폴백

**요청**: 동일, Vision 감지 실패

**기대 결과**:
- ✅ judge_by_weight_only() 호출
- ✅ has_loadcell="false" 상품 제외
- ✅ 로그: `[LOADCELL-ONLY] v4.8: Skip no-loadcell product...`
- ✅ 다른 로드셀 상품에서만 매칭 시도

---

## 로깅 개선

### count_calculator.py 로그

```
[COUNT] v4.8: Using active_product: {product_name} (class_id={class_id}, weight={weight}g, stock={stock}, has_loadcell={has_loadcell})
[COUNT] v4.8: Stock depleted: {product_name} (stock=0, vision detected but filtered)
[COUNT] v4.8: Vision-only mode: {product_name} (has_loadcell=false)
```

### decision_engine.py 로그

```
[LOADCELL-ONLY] v4.8: Using {count} products from active_products (Node.js latest weights)
[LOADCELL-ONLY] v4.8: Using {count} products from ProductDB (fallback)
[LOADCELL-ONLY] v4.8: Skip no-loadcell product: {product_name}
```

### judgment_service.py 로그

```
[SYNC] Product {p.product_idx} (ID:{class_id}) updated: weight={weight}g, stock={p.stock_qty}, has_loadcell={p.has_loadcell}
```

---

## 하위 호환성

- **has_loadcell 기본값**: Node.js가 보내지 않으면 "true" (로드셀 있음)
- **ProductDB 기존 상품**: "true"로 간주
- **필드 없으면**: 기존 로직(로드셀 사용) 유지

---

## 배포 전 체크리스트

- [x] 모든 수정 파일 코드 리뷰
- [x] 로깅 메시지 검증 (v4.8 태그 확인)
- [ ] 단위 테스트 실행 (pytest)
- [ ] 통합 테스트 시나리오 실행
- [ ] Node.js 연동 테스트 (실제 환경)
- [ ] CLAUDE.md TODO 섹션에서 has_loadcell 항목 제거
- [ ] 버전 번호 업데이트 (v4.7 → v4.8)
- [ ] 변경 로그 작성

---

## 참고 문서

- Edge_Environment/CLAUDE.md - TODO 섹션 has_loadcell 필드 지원
- 계획 문서 (이 파일 작성 기반)
