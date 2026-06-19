"""
Active Product Store (v4.5).

Node.js에서 받은 활성 상품 정보를 전역(global)으로 관리.
YOLO classes 파라미터를 통한 사전 필터링 지원.

v4.5 변경사항:
- zone별 관리 → 전역 관리로 변경
- set_products()에서 zone 파라미터 제거
- has_products(), get_allowed_class_ids(), clear()에서 zone 파라미터 제거
- get_product_by_class_id(), clear_all() 메서드 제거

핵심 기능:
- Node.js products 배열에서 YOLO class_id 추출
- stock_qty > 0 상품만 allowed_class_ids에 포함
- 문 닫힘(finalize) 시 자동 정리

사용법:
    store = ActiveProductStore(yolo_mapping=yolo_mapping_dict)

    # products 설정 (Node.js 폴링 시)
    result = store.set_products(products=products_list)

    # YOLO 추론 전 허용 클래스 조회
    allowed_ids = store.get_allowed_class_ids()

    # 문 닫힘 시 정리
    store.clear()
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CATALOG_SOURCE_POLICIES = {"node_first", "static_mapping_compat"}
DIRECT_CLASS_ID_FIELDS = (
    "yolo_class_id",
    "yoloClassId",
    "trainingidx",
    "training_idx",
    "trainingIdx",
)

YOLO_NAME_ALIASES: Dict[str, str] = {
    "BAG_JAYEONRU_MOIST_SWEET_CHESTNUT_80G": "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G",
    "BAG_JAYEONLU_MOIST_SWEET_CHESTNUT_80G": "BAG_JAYEONRU_MOIST_SWEET_CHESTNUT_80G",
}

PRODUCT_WEIGHT_FIELD_ALIASES = (
    "product_weight",
    "productWeight",
    "product_weight_g",
    "productWeightG",
    "unit_weight_g",
    "unitWeightG",
    "weight",
)

KNOWN_PRODUCT_WEIGHT_FALLBACKS: Dict[int, float] = {
    44: 520.0,  # BOTTLE_KWANGDONG_CORN_SILK_TEA_500ML
}


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
    class_id_source: str = "unknown"

    # v5.0: product_db 호환 프로퍼티 (decision_engine 등에서 사용)
    @property
    def product_id(self) -> int:
        return self.yolo_class_id or 0

    @property
    def weight(self) -> float:
        return self.product_weight

    @property
    def price(self) -> int:
        return self.sale_price

    @property
    def name(self) -> str:
        return self.product_name

    @property
    def stock(self) -> int:
        return self.stock_qty


@dataclass
class GlobalProductData:
    """전역 상품 데이터 (v4.5)."""

    products: List[ProductInfo]  # Node.js 원본 (매핑 성공한 것만)
    allowed_class_ids: List[int]  # YOLO 허용 클래스 IDs (stock > 0)
    class_to_product: Dict[int, ProductInfo]  # class_id -> ProductInfo 빠른 조회
    zero_stock_products: int = 0
    stock_positive_class_products: int = 0
    stock_positive_weight_products: int = 0
    weight_unavailable_products: int = 0
    repaired_weight_products: int = 0
    repaired_weight_diagnostics: List[dict] = field(default_factory=list)
    unmapped_total: int = 0
    unmapped_names: List[str] = field(default_factory=list)
    invalid_class_id_total: int = 0
    invalid_class_ids: List[dict] = field(default_factory=list)
    catalog_source_policy: str = "node_first"
    updated_at: float = field(default_factory=time.time)


@dataclass
class SetProductsResult:
    """set_products() 결과."""

    success: bool
    total_products: int  # Node.js에서 받은 총 상품 수
    mapped_products: int  # YOLO 매핑 성공한 상품 수
    allowed_products: int  # stock > 0인 상품 수 (allowed_class_ids 개수)
    unmapped_names: List[str]  # 매핑 실패한 상품명 목록
    zero_stock_products: int = 0
    stock_positive_class_products: int = 0
    stock_positive_weight_products: int = 0
    weight_unavailable_products: int = 0
    repaired_weight_products: int = 0
    repaired_weight_diagnostics: List[dict] = field(default_factory=list)
    unmapped_total: int = 0
    preserved_existing: bool = False
    invalid_class_id_total: int = 0
    invalid_class_ids: List[dict] = field(default_factory=list)


@dataclass
class EffectiveProductSnapshot:
    """Active-product snapshot selected for inference."""

    products: List[ProductInfo]
    allowed_class_ids: Optional[List[int]]
    source: str
    age_seconds: Optional[float] = None
    used_last_valid_snapshot: bool = False
    current_snapshot_present: bool = False
    last_valid_snapshot_present: bool = False
    last_valid_snapshot_expired: bool = False

    def diagnostics(self) -> dict:
        diagnostics = {
            "snapshot_source": self.source,
            "used_last_valid_snapshot": self.used_last_valid_snapshot,
            "current_snapshot_present": self.current_snapshot_present,
            "last_valid_snapshot_present": self.last_valid_snapshot_present,
            "last_valid_snapshot_expired": self.last_valid_snapshot_expired,
        }
        if self.age_seconds is not None:
            diagnostics["last_valid_age_seconds"] = round(self.age_seconds, 3)
        return diagnostics


class ActiveProductStore:
    """
    Node.js에서 받은 활성 상품 정보를 전역(global)으로 관리 (v4.5).

    YOLO classes 파라미터를 통한 사전 필터링 지원.
    yolo_product_mapping.json에서 로드된 매핑 정보를 사용하여
    product_name → yolo_class_id 변환.

    Thread-safe: 내부 Lock 사용.
    """

    def __init__(
        self,
        yolo_name_to_id: Optional[Dict[str, int]] = None,
        last_valid_ttl_seconds: float = 300.0,
        source_policy: str = "node_first",
    ):
        """
        Initialize ActiveProductStore.

        Args:
            yolo_name_to_id: {yolo_class_name: yolo_class_id} 매핑
                             yolo_product_mapping.json에서 로드
        """
        normalized_policy = str(source_policy).strip().lower()
        if normalized_policy not in CATALOG_SOURCE_POLICIES:
            raise ValueError(f"Invalid catalog source policy: {source_policy}")
        self.source_policy = normalized_policy
        self._engine_name_to_id: Dict[str, int] = dict(yolo_name_to_id or {})
        self._yolo_name_to_id: Dict[str, int] = dict(self._engine_name_to_id)
        self._yolo_name_normalized: Dict[str, int] = {}  # 정규화된 이름 매핑
        self._known_class_ids: set[int] = set(self._engine_name_to_id.values())
        self._global_data: Optional[GlobalProductData] = None  # 전역 상품 데이터 (v4.5)
        self._last_valid_data: Optional[GlobalProductData] = None
        self._last_valid_ttl_seconds = max(0.0, float(last_valid_ttl_seconds))
        self._lock = threading.Lock()

        # 정규화된 이름 매핑 생성
        self._build_normalized_mapping()

        logger.info(
            f"ActiveProductStore initialized: "
            f"{len(self._engine_name_to_id)} YOLO classes, "
            f"source_policy={self.source_policy}"
        )

    def _build_normalized_mapping(self) -> None:
        """정규화된 이름 매핑 생성."""
        for name, class_id in list(self._engine_name_to_id.items()):
            self._register_yolo_name(name, class_id, engine_backed=True)
        self._register_aliases()

    def _register_yolo_name(
        self,
        name: str,
        class_id: int,
        *,
        engine_backed: bool = False,
    ) -> None:
        self._yolo_name_to_id[name] = class_id
        self._yolo_name_normalized[self._normalize_name(name)] = class_id
        if engine_backed:
            self._known_class_ids.add(class_id)

    def _register_aliases(self) -> None:
        for alias, canonical in YOLO_NAME_ALIASES.items():
            alias_key = self._normalize_name(alias)
            canonical_key = self._normalize_name(canonical)
            class_id = self._yolo_name_normalized.get(canonical_key)
            if class_id is None:
                class_id = self._yolo_name_normalized.get(alias_key)
            if class_id is not None:
                self._yolo_name_normalized[alias_key] = class_id
                self._yolo_name_normalized[canonical_key] = class_id

    @staticmethod
    def _normalize_name(name: str) -> str:
        """
        상품명 정규화.

        밑줄, 공백, 하이픈 등 제거하고 대문자로 변환.

        Args:
            name: 상품명

        Returns:
            정규화된 이름
        """
        # 밑줄, 하이픈, 공백, 점, 슬래시 등 제거
        normalized = re.sub(r'[_\-\.\s\/\\]', '', name)
        return normalized.upper()

    def load_yolo_mapping(self, mapping_data: dict) -> int:
        """
        yolo_product_mapping.json 데이터 로드.

        Args:
            mapping_data: {"mappings": [{"yolo_class_id": 1, "yolo_class_name": "..."}, ...]}

        Returns:
            로드된 매핑 수
        """
        if self.source_policy != "static_mapping_compat":
            logger.info(
                "[ActiveProductStore] skipped static YOLO mapping: "
                f"source_policy={self.source_policy}"
            )
            return 0

        mappings = mapping_data.get("mappings", [])
        count = 0

        for m in mappings:
            yolo_class_id = m.get("yolo_class_id")
            yolo_class_name = m.get("yolo_class_name")

            if yolo_class_id is None or not yolo_class_name:
                continue

            self._register_yolo_name(yolo_class_name, int(yolo_class_id))
            count += 1

        self._register_aliases()
        logger.info(f"Loaded {count} YOLO class mappings")
        return count

    def _find_yolo_class_id(self, product_name: str) -> Optional[int]:
        """
        상품명으로 YOLO class_id 찾기.

        매칭 순서:
        1. 정확한 이름 매칭 (대소문자 무시)
        2. 정규화 후 매칭 (밑줄/공백 무시)

        Args:
            product_name: Node.js에서 받은 상품명

        Returns:
            YOLO class_id 또는 None
        """
        if not product_name:
            return None

        # 1. 정확한 이름 매칭 (대소문자 무시)
        name_upper = product_name.upper()
        for yolo_name, class_id in self._yolo_name_to_id.items():
            if yolo_name.upper() == name_upper:
                return class_id

        # 2. 정규화 후 매칭
        normalized = self._normalize_name(product_name)
        return self._yolo_name_normalized.get(normalized)

    def _coerce_yolo_class_id(self, value: object) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            class_id = int(value)
        except (TypeError, ValueError):
            return None
        if class_id <= 0:
            return None
        if self._known_class_ids and class_id not in self._known_class_ids:
            return None
        return class_id

    def _resolve_yolo_class_id(
        self,
        product: dict,
    ) -> tuple[Optional[int], str, Optional[dict]]:
        for field_name in DIRECT_CLASS_ID_FIELDS:
            if field_name not in product:
                continue
            raw_value = product.get(field_name)
            direct_id = self._coerce_yolo_class_id(raw_value)
            if direct_id is not None:
                return direct_id, field_name, None
            if raw_value not in (None, ""):
                return None, field_name, {
                    "source": field_name,
                    "value": raw_value,
                }

        class_name = product.get("yolo_class_name", product.get("yoloClassName"))
        if class_name:
            class_id = self._find_yolo_class_id(str(class_name))
            if class_id is not None:
                resolved = self._coerce_yolo_class_id(class_id)
                if resolved is not None:
                    return resolved, "yolo_class_name", None
                return None, "yolo_class_name", {
                    "source": "yolo_class_name",
                    "value": class_name,
                    "resolved_class_id": class_id,
                }

        if self.source_policy == "static_mapping_compat":
            class_id = self._find_yolo_class_id(product.get("product_name", ""))
            if class_id is not None:
                resolved = self._coerce_yolo_class_id(class_id)
                if resolved is not None:
                    return resolved, "product_name_static_mapping", None
                return None, "product_name_static_mapping", {
                    "source": "product_name_static_mapping",
                    "value": product.get("product_name", ""),
                    "resolved_class_id": class_id,
                }

        return None, "unmapped", None

    @staticmethod
    def _is_stock_positive_weight_product(product: ProductInfo) -> bool:
        return product.stock_qty > 0 and product.product_weight > 0

    @staticmethod
    def _is_stock_positive_class_product(product: ProductInfo) -> bool:
        return product.stock_qty > 0 and product.yolo_class_id is not None

    @staticmethod
    def _coerce_product_weight(value: object) -> float:
        try:
            if value in (None, ""):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _extract_product_weight(cls, product: dict) -> tuple[float, Optional[dict]]:
        primary_value = product.get("product_weight")
        primary_weight = cls._coerce_product_weight(primary_value)
        if primary_weight > 0:
            return primary_weight, None

        for field_name in PRODUCT_WEIGHT_FIELD_ALIASES:
            if field_name == "product_weight" or field_name not in product:
                continue
            alias_weight = cls._coerce_product_weight(product.get(field_name))
            if alias_weight <= 0:
                continue
            return alias_weight, {
                "source": "payload_alias",
                "field": field_name,
                "original_weight": primary_value,
                "repaired_weight": alias_weight,
            }

        return primary_weight, None

    @staticmethod
    def _product_maps_by_idx(products: Dict[int, ProductInfo]) -> Dict[str, ProductInfo]:
        return {
            product.product_idx: product
            for product in products.values()
            if product.product_idx
        }

    @staticmethod
    def _positive_weight_from_snapshot(
        *,
        yolo_class_id: int,
        product_idx: str,
        products_by_class: Dict[int, ProductInfo],
        products_by_idx: Dict[str, ProductInfo],
    ) -> float:
        product = products_by_class.get(yolo_class_id)
        if product is not None and product.product_weight > 0:
            return product.product_weight
        if product_idx:
            product = products_by_idx.get(product_idx)
            if product is not None and product.product_weight > 0:
                return product.product_weight
        return 0.0

    def _known_product_weight_fallback(
        self,
        *,
        yolo_class_id: int,
        product_name: str,
    ) -> float:
        if yolo_class_id in KNOWN_PRODUCT_WEIGHT_FALLBACKS:
            return KNOWN_PRODUCT_WEIGHT_FALLBACKS[yolo_class_id]
        normalized_name = self._normalize_name(product_name)
        class_id = self._yolo_name_normalized.get(normalized_name)
        if class_id in KNOWN_PRODUCT_WEIGHT_FALLBACKS:
            return KNOWN_PRODUCT_WEIGHT_FALLBACKS[class_id]
        return 0.0

    def _repair_product_weight(
        self,
        *,
        product_name: str,
        product_idx: str,
        yolo_class_id: int,
        product_weight: float,
        original_weight: object,
        stock_qty: int,
        current_by_class: Dict[int, ProductInfo],
        current_by_idx: Dict[str, ProductInfo],
        last_valid_by_class: Dict[int, ProductInfo],
        last_valid_by_idx: Dict[str, ProductInfo],
    ) -> tuple[float, Optional[dict]]:
        if product_weight > 0 or stock_qty <= 0:
            return product_weight, None

        repaired_weight = self._positive_weight_from_snapshot(
            yolo_class_id=yolo_class_id,
            product_idx=product_idx,
            products_by_class=current_by_class,
            products_by_idx=current_by_idx,
        )
        source = "current_snapshot"

        if repaired_weight <= 0:
            repaired_weight = self._positive_weight_from_snapshot(
                yolo_class_id=yolo_class_id,
                product_idx=product_idx,
                products_by_class=last_valid_by_class,
                products_by_idx=last_valid_by_idx,
            )
            source = "last_valid_snapshot"

        if repaired_weight <= 0 and self.source_policy == "static_mapping_compat":
            repaired_weight = self._known_product_weight_fallback(
                yolo_class_id=yolo_class_id,
                product_name=product_name,
            )
            source = "known_product_weight_fallback"

        if repaired_weight <= 0:
            return product_weight, None

        return repaired_weight, {
            "source": source,
            "field": "product_weight",
            "product_idx": product_idx,
            "product_name": product_name,
            "class_id": yolo_class_id,
            "original_weight": original_weight,
            "repaired_weight": repaired_weight,
        }

    def set_products(
        self,
        products: List[dict],
        *,
        preserve_on_invalid_existing: bool = False,
    ) -> SetProductsResult:
        """
        전역 상품 리스트 저장 (v4.5).

        product_name → yolo_class_id 매핑 수행.
        stock_qty > 0인 상품만 allowed_class_ids에 추가.

        Args:
            products: Node.js에서 받은 상품 리스트
                      [{"product_idx": "...", "product_name": "...",
                        "sale_price": 0, "product_weight": "0", "stock_qty": 0}, ...]

        Returns:
            SetProductsResult
        """
        mapped_products: List[ProductInfo] = []
        allowed_class_ids: List[int] = []
        class_to_product: Dict[int, ProductInfo] = {}
        unmapped_names: List[str] = []
        invalid_class_ids: List[dict] = []
        zero_stock_products = 0
        stock_positive_class_products = 0
        stock_positive_weight_products = 0
        weight_unavailable_products = 0
        repaired_weight_diagnostics: List[dict] = []

        with self._lock:
            current_by_class = (
                dict(self._global_data.class_to_product)
                if self._global_data is not None
                else {}
            )
            current_by_idx = self._product_maps_by_idx(current_by_class)
            last_valid_by_class = (
                dict(self._last_valid_data.class_to_product)
                if self._last_valid_is_fresh_locked()
                else {}
            )
            last_valid_by_idx = self._product_maps_by_idx(last_valid_by_class)

        for p in products:
            product_name = p.get("product_name", "")
            product_idx = p.get("product_idx", "")
            sale_price = int(p.get("sale_price", 0))
            stock_raw = p.get("stock_qty", 0)
            stock_qty = 999 if stock_raw is None else int(stock_raw)
            has_loadcell = p.get("has_loadcell", "true")  # v4.8: 추가

            # product_weight는 문자열일 수 있음
            weight_str = p.get("product_weight", "0")
            product_weight, weight_repair = self._extract_product_weight(p)

            # YOLO class_id 찾기
            yolo_class_id, class_id_source, invalid_class_id = (
                self._resolve_yolo_class_id(p)
            )

            if yolo_class_id is None:
                unmapped_names.append(product_name)
                if invalid_class_id is not None:
                    invalid_class_ids.append(
                        {
                            "product_idx": product_idx,
                            "product_name": product_name,
                            **invalid_class_id,
                        }
                    )
                continue

            if weight_repair is None:
                product_weight, weight_repair = self._repair_product_weight(
                    product_name=product_name,
                    product_idx=product_idx,
                    yolo_class_id=yolo_class_id,
                    product_weight=product_weight,
                    original_weight=weight_str,
                    stock_qty=stock_qty,
                    current_by_class=current_by_class,
                    current_by_idx=current_by_idx,
                    last_valid_by_class=last_valid_by_class,
                    last_valid_by_idx=last_valid_by_idx,
                )
            else:
                weight_repair.update(
                    {
                        "product_idx": product_idx,
                        "product_name": product_name,
                        "class_id": yolo_class_id,
                    }
                )
            if weight_repair is not None:
                repaired_weight_diagnostics.append(weight_repair)

            # ProductInfo 생성
            product_info = ProductInfo(
                product_idx=product_idx,
                product_name=product_name,
                sale_price=sale_price,
                product_weight=product_weight,
                stock_qty=stock_qty,
                yolo_class_id=yolo_class_id,
                has_loadcell=has_loadcell,  # v4.8: 추가
                class_id_source=class_id_source,
            )
            mapped_products.append(product_info)
            class_to_product[yolo_class_id] = product_info
            if self._is_stock_positive_class_product(product_info):
                stock_positive_class_products += 1
                if product_info.product_weight <= 0:
                    weight_unavailable_products += 1
            if self._is_stock_positive_weight_product(product_info):
                stock_positive_weight_products += 1

            # stock > 0이면 (또는 None이면) allowed_class_ids에 추가 (v4.6)
            if stock_qty is None or stock_qty > 0:
                allowed_class_ids.append(yolo_class_id)
                logger.debug(f"[ActiveProductStore] Allowed: {product_name} (stock={stock_qty})")
            else:
                zero_stock_products += 1
                logger.warning(f"[ActiveProductStore] Filtered out: {product_name} (stock={stock_qty})")

        # 전역 데이터 저장 (v4.5)
        preserved_existing = False
        with self._lock:
            existing_valid_count = (
                self._global_data.stock_positive_class_products
                if self._global_data is not None
                else 0
            )
            if (
                preserve_on_invalid_existing
                and existing_valid_count > 0
                and stock_positive_class_products == 0
            ):
                preserved_existing = True
            else:
                new_data = GlobalProductData(
                    products=mapped_products,
                    allowed_class_ids=allowed_class_ids,
                    class_to_product=class_to_product,
                    zero_stock_products=zero_stock_products,
                    stock_positive_class_products=stock_positive_class_products,
                    stock_positive_weight_products=stock_positive_weight_products,
                    weight_unavailable_products=weight_unavailable_products,
                    repaired_weight_products=len(repaired_weight_diagnostics),
                    repaired_weight_diagnostics=repaired_weight_diagnostics[:10],
                    unmapped_total=len(unmapped_names),
                    unmapped_names=unmapped_names[:10],
                    invalid_class_id_total=len(invalid_class_ids),
                    invalid_class_ids=invalid_class_ids[:10],
                    catalog_source_policy=self.source_policy,
                    updated_at=time.time(),
                )
                self._global_data = new_data
                if stock_positive_class_products > 0:
                    self._last_valid_data = new_data

        result = SetProductsResult(
            success=len(mapped_products) > 0,
            total_products=len(products),
            mapped_products=len(mapped_products),
            allowed_products=len(allowed_class_ids),
            unmapped_names=unmapped_names[:10],
            zero_stock_products=zero_stock_products,
            stock_positive_class_products=stock_positive_class_products,
            stock_positive_weight_products=stock_positive_weight_products,
            weight_unavailable_products=weight_unavailable_products,
            repaired_weight_products=len(repaired_weight_diagnostics),
            repaired_weight_diagnostics=repaired_weight_diagnostics[:10],
            unmapped_total=len(unmapped_names),
            preserved_existing=preserved_existing,
            invalid_class_id_total=len(invalid_class_ids),
            invalid_class_ids=invalid_class_ids[:10],
        )

        if preserved_existing:
            logger.warning(
                "[ActiveProductStore] global: ignored invalid product snapshot "
                f"(mapped={result.mapped_products}/{result.total_products}, "
                f"allowed_classes={result.allowed_products}, "
                f"stock_positive_class={result.stock_positive_class_products}, "
                f"stock_positive_weight={result.stock_positive_weight_products}); "
                f"preserved_existing_stock_positive_class={existing_valid_count}"
            )
        else:
            logger.info(
                f"[ActiveProductStore] global: "
                f"set {result.mapped_products}/{result.total_products} products, "
                f"allowed_classes={len(allowed_class_ids)}, "
                f"stock_positive_class={stock_positive_class_products}, "
                f"stock_positive_weight={stock_positive_weight_products}, "
                f"weight_unavailable={weight_unavailable_products}, "
                f"repaired_weights={result.repaired_weight_products}"
            )

        if unmapped_names:
            logger.warning(
                f"[ActiveProductStore] global: "
                f"{len(unmapped_names)} unmapped products: {unmapped_names[:5]}"
            )

        if repaired_weight_diagnostics:
            logger.warning(
                "[ActiveProductStore] repaired product weights: "
                f"{repaired_weight_diagnostics[:5]}"
            )

        if invalid_class_ids:
            logger.warning(
                "[ActiveProductStore] invalid class ids: "
                f"{invalid_class_ids[:5]}"
            )

        return result

    def has_products(self) -> bool:
        """
        전역 상품 정보가 있는지 확인 (v4.5).

        Returns:
            상품 정보 존재 여부
        """
        with self._lock:
            return self._global_data is not None and len(self._global_data.products) > 0

    def get_allowed_class_ids(self) -> Optional[List[int]]:
        """
        허용된 YOLO 클래스 ID 리스트 (v4.5).

        stock_qty > 0인 상품의 YOLO class_id만 반환.

        Returns:
            허용된 class_id 리스트.
            None이면 상품 정보가 없음 (YOLO 전체 클래스 탐지).
            빈 리스트면 허용된 상품이 없음 (모두 재고 0).
        """
        with self._lock:
            if self._global_data is None:
                return None
            return self._global_data.allowed_class_ids.copy()

    def get_all_products(self) -> List[ProductInfo]:
        """
        모든 상품 정보 조회 (v4.5).

        Returns:
            ProductInfo 리스트 (빈 리스트면 없음)
        """
        with self._lock:
            if self._global_data is None:
                return []
            return self._global_data.products.copy()

    def has_stock_positive_weight_products(self) -> bool:
        """Return True when the current snapshot has usable inventory rows."""
        with self._lock:
            return (
                self._global_data is not None
                and self._global_data.stock_positive_weight_products > 0
            )

    def has_stock_positive_class_products(self) -> bool:
        """Return True when the current snapshot has stock-positive class rows."""
        with self._lock:
            return (
                self._global_data is not None
                and self._global_data.stock_positive_class_products > 0
            )

    def _last_valid_age_locked(self, now: Optional[float] = None) -> Optional[float]:
        if self._last_valid_data is None:
            return None
        now = time.time() if now is None else now
        return max(0.0, now - self._last_valid_data.updated_at)

    def _last_valid_is_fresh_locked(self, now: Optional[float] = None) -> bool:
        age = self._last_valid_age_locked(now)
        if age is None:
            return False
        return age <= self._last_valid_ttl_seconds

    @staticmethod
    def _copy_snapshot(
        data: Optional[GlobalProductData],
    ) -> tuple[List[ProductInfo], Optional[List[int]]]:
        if data is None:
            return [], None
        return data.products.copy(), data.allowed_class_ids.copy()

    def get_effective_snapshot(self) -> EffectiveProductSnapshot:
        """
        Return the inference snapshot.

        The current snapshot wins when it has stock-positive, weight-valid rows.
        If CLOSE cleared current data or an invalid payload arrived, a fresh
        last-valid snapshot is used as a bounded fallback.
        """
        with self._lock:
            now = time.time()
            current = self._global_data
            last_valid = self._last_valid_data
            last_valid_age = self._last_valid_age_locked(now)
            last_valid_fresh = self._last_valid_is_fresh_locked(now)
            last_valid_expired = last_valid is not None and not last_valid_fresh

            if current is not None and current.stock_positive_class_products > 0:
                products, allowed_class_ids = self._copy_snapshot(current)
                return EffectiveProductSnapshot(
                    products=products,
                    allowed_class_ids=allowed_class_ids,
                    source="current",
                    age_seconds=last_valid_age,
                    current_snapshot_present=True,
                    last_valid_snapshot_present=last_valid is not None,
                    last_valid_snapshot_expired=last_valid_expired,
                )

            if last_valid is not None and last_valid_fresh:
                products, allowed_class_ids = self._copy_snapshot(last_valid)
                return EffectiveProductSnapshot(
                    products=products,
                    allowed_class_ids=allowed_class_ids,
                    source="last_valid",
                    age_seconds=last_valid_age,
                    used_last_valid_snapshot=True,
                    current_snapshot_present=current is not None,
                    last_valid_snapshot_present=True,
                    last_valid_snapshot_expired=False,
                )

            if current is not None:
                products, allowed_class_ids = self._copy_snapshot(current)
                return EffectiveProductSnapshot(
                    products=products,
                    allowed_class_ids=allowed_class_ids,
                    source="current",
                    age_seconds=last_valid_age,
                    current_snapshot_present=True,
                    last_valid_snapshot_present=last_valid is not None,
                    last_valid_snapshot_expired=last_valid_expired,
                )

            return EffectiveProductSnapshot(
                products=[],
                allowed_class_ids=None,
                source="missing",
                age_seconds=last_valid_age,
                current_snapshot_present=False,
                last_valid_snapshot_present=last_valid is not None,
                last_valid_snapshot_expired=last_valid_expired,
            )

    def get_effective_products(self) -> List[ProductInfo]:
        """Return products from the current or fresh last-valid snapshot."""
        return self.get_effective_snapshot().products

    def get_effective_allowed_class_ids(self) -> Optional[List[int]]:
        """Return allowed classes from the current or fresh last-valid snapshot."""
        return self.get_effective_snapshot().allowed_class_ids

    def clear(self, *, preserve_last_valid: bool = True) -> bool:
        """
        전역 상품 데이터 삭제 (v4.5).

        Returns:
            삭제 성공 여부 (데이터가 있었으면 True)
        """
        with self._lock:
            had_current = self._global_data is not None
            had_last_valid = self._last_valid_data is not None
            self._global_data = None
            if not preserve_last_valid:
                self._last_valid_data = None
            if had_current or (had_last_valid and not preserve_last_valid):
                suffix = "preserved_last_valid" if preserve_last_valid else "cleared_all"
                logger.info(f"[ActiveProductStore] global: cleared ({suffix})")
                return True
            return False

    def clear_all(self) -> bool:
        """Clear current and last-valid snapshots."""
        return self.clear(preserve_last_valid=False)

    # ========================================================================
    # v5.0: product_db 호환 메서드 (ProductDatabase 대체)
    # ========================================================================

    def get_by_yolo_class_id(self, class_id: int) -> Optional[ProductInfo]:
        """yolo_class_id로 상품 조회 (product_db.get_by_yolo_class_id 대체)."""
        with self._lock:
            if self._global_data is not None:
                product = self._global_data.class_to_product.get(class_id)
                if product is not None:
                    return product
            if self._last_valid_is_fresh_locked():
                return self._last_valid_data.class_to_product.get(class_id)  # type: ignore[union-attr]
            return None

    def get_by_yolo_class_name(self, class_name: str) -> Optional[ProductInfo]:
        """yolo_class_name으로 상품 조회 (product_db.get_by_yolo_class_name 대체)."""
        class_id = self._find_yolo_class_id(class_name)
        if class_id is None:
            return None
        return self.get_by_yolo_class_id(class_id)

    def get_product_weight(self, class_id: int) -> float:
        """상품 무게 조회 (product_db.get_weight 대체)."""
        product = self.get_by_yolo_class_id(class_id)
        return product.product_weight if product else 0.0

    def get_product_price(self, class_id: int) -> int:
        """상품 가격 조회 (product_db.get_price 대체)."""
        product = self.get_by_yolo_class_id(class_id)
        return product.sale_price if product else 0

    def get_product(self, class_id: int) -> Optional[ProductInfo]:
        """product_id(class_id)로 상품 조회 (product_db.get_product 대체)."""
        return self.get_by_yolo_class_id(class_id)

    def get_price(self, class_id: int) -> int:
        """상품 가격 조회 (product_db.get_price 호환)."""
        return self.get_product_price(class_id)

    def get_stats(self) -> dict:
        """
        저장소 통계 반환 (v4.5).

        Returns:
            통계 정보
        """
        with self._lock:
            stats = {
                "total_yolo_classes": len(self._engine_name_to_id),
                "catalog_source_policy": self.source_policy,
                "has_products": self._global_data is not None,
                "has_last_valid_snapshot": self._last_valid_data is not None,
            }

            last_valid_age = self._last_valid_age_locked()
            if last_valid_age is not None:
                stats["last_valid_age_seconds"] = round(last_valid_age, 3)
                stats["last_valid_ttl_seconds"] = self._last_valid_ttl_seconds
                stats["last_valid_expired"] = (
                    last_valid_age > self._last_valid_ttl_seconds
                )

            if self._global_data is not None:
                stats["products_count"] = len(self._global_data.products)
                stats["allowed_classes_count"] = len(self._global_data.allowed_class_ids)
                stats["zero_stock_products"] = self._global_data.zero_stock_products
                stats["stock_positive_class_products"] = (
                    self._global_data.stock_positive_class_products
                )
                stats["stock_positive_weight_products"] = (
                    self._global_data.stock_positive_weight_products
                )
                stats["weight_unavailable_products"] = (
                    self._global_data.weight_unavailable_products
                )
                stats["repaired_weight_products"] = (
                    self._global_data.repaired_weight_products
                )
                stats["repaired_weight_diagnostics"] = list(
                    self._global_data.repaired_weight_diagnostics
                )
                stats["unmapped_products"] = self._global_data.unmapped_total
                stats["unmapped_names"] = list(self._global_data.unmapped_names)
                stats["invalid_class_id_products"] = (
                    self._global_data.invalid_class_id_total
                )
                stats["invalid_class_ids"] = list(self._global_data.invalid_class_ids)
                stats["updated_at"] = self._global_data.updated_at

            return stats
