"""
크로스 존 반환 테스트 (v4.9).

Zone A에서 꺼낸 상품을 Zone B에 넣으면 Zone A에서 차감하는 로직 테스트.
"""

import time
import tempfile
import pytest

from model_service.session import ProductResult, DoorSessionStore
from model_service.session.door_session import TriggerResult


class TestCrossZoneReturn:
    """크로스 존 반환 처리 테스트."""

    @pytest.fixture
    def store(self):
        """테스트용 DoorSessionStore 생성."""

        def get_weight(product_id: int) -> float:
            weights = {26: 365.0, 27: 250.0, 28: 400.0}
            return weights.get(product_id, 0.0)

        temp_dir = tempfile.mkdtemp()
        store = DoorSessionStore(
            yaml_dir=temp_dir,
            session_timeout=30.0,
            weight_tolerance=5.0,
            get_product_weight=get_weight,
        )
        yield store
        store.clear_all()

    def _create_removal_trigger(
        self, zone: int, delta: float, product_id: int, name: str, count: int
    ) -> TriggerResult:
        """제거 트리거 생성 (delta < 0)."""
        return TriggerResult(
            trigger_id="",
            session_id=f"zone_{zone}_{int(time.time())}",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=count,
                    price=3500,
                    confidence=0.95,
                )
            ],
            delta_weight=delta,
            confidence=0.95,
            video_paths={},
            is_return=False,
        )

    def _create_return_trigger(self, zone: int, delta: float) -> TriggerResult:
        """반환 트리거 생성 (delta > 0, 상품 없음)."""
        return TriggerResult(
            trigger_id="",
            session_id=f"zone_{zone}_{int(time.time())}",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    def test_cross_zone_return_basic(self, store):
        """Zone 1에서 꺼내고 Zone 2에서 반환 시 Zone 1에서 차감."""
        # Zone 1: 치킨마요 2개 꺼냄 (delta=-730g)
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-730.0, product_id=26, name="치킨마요", count=2
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 치킨마요 1개 반환 (delta=+365g)
        trigger2 = self._create_return_trigger(zone=2, delta=365.0)
        store.add_trigger(zone=2, result=trigger2)

        # 확인: Zone 1의 치킨마요가 1개로 차감
        zone1_session = store.get_session(zone=1)
        assert zone1_session is not None
        assert zone1_session.aggregated_products[26].count == 1

        # 확인: Zone 2에 cross_zone_returns 기록
        zone2_session = store.get_session(zone=2)
        assert zone2_session is not None
        assert len(zone2_session.cross_zone_returns) == 1
        assert zone2_session.cross_zone_returns[0].target_zone == 1
        assert zone2_session.cross_zone_returns[0].product_name == "치킨마요"
        assert zone2_session.cross_zone_returns[0].source_zone == 2

    def test_same_zone_return_first(self, store):
        """같은 zone 반환은 같은 zone에서 먼저 처리."""
        # Zone 1: 치킨마요 2개 꺼냄
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-730.0, product_id=26, name="치킨마요", count=2
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 1: 치킨마요 1개 반환 (같은 zone)
        trigger2 = self._create_return_trigger(zone=1, delta=365.0)
        store.add_trigger(zone=1, result=trigger2)

        # 확인: Zone 1에서 처리됨 (크로스 존 아님)
        zone1_session = store.get_session(zone=1)
        assert zone1_session is not None
        assert zone1_session.aggregated_products[26].count == 1
        assert len(zone1_session.cross_zone_returns) == 0

    def test_cross_zone_no_match(self, store):
        """매칭 상품 없으면 unmatched로 유지."""
        # Zone 1: 참치마요 1개 꺼냄 (250g)
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-250.0, product_id=27, name="참치마요", count=1
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 400g 반환 (매칭 상품 없음 - 참치마요는 250g)
        trigger2 = self._create_return_trigger(zone=2, delta=400.0)
        store.add_trigger(zone=2, result=trigger2)

        # 확인: unmatched로 유지
        zone2_session = store.get_session(zone=2)
        assert zone2_session is not None
        assert len(zone2_session.unmatched_returns) == 1
        assert len(zone2_session.cross_zone_returns) == 0

        # Zone 1의 참치마요는 그대로
        zone1_session = store.get_session(zone=1)
        assert zone1_session is not None
        assert zone1_session.aggregated_products[27].count == 1

    def test_multiple_zones_cross_return(self, store):
        """여러 zone 간 크로스 반환."""
        # Zone 1: 치킨마요 1개 (365g)
        store.add_trigger(
            zone=1,
            result=self._create_removal_trigger(
                zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
            ),
        )

        # Zone 2: 참치마요 1개 (250g)
        store.add_trigger(
            zone=2,
            result=self._create_removal_trigger(
                zone=2, delta=-250.0, product_id=27, name="참치마요", count=1
            ),
        )

        # Zone 3: 치킨마요 반환 (365g) -> Zone 1에서 차감
        store.add_trigger(zone=3, result=self._create_return_trigger(zone=3, delta=365.0))

        # Zone 3: 참치마요 반환 (250g) -> Zone 2에서 차감
        store.add_trigger(zone=3, result=self._create_return_trigger(zone=3, delta=250.0))

        # 확인
        zone1_session = store.get_session(zone=1)
        zone2_session = store.get_session(zone=2)
        zone3_session = store.get_session(zone=3)

        assert zone1_session.aggregated_products[26].count == 0
        assert zone2_session.aggregated_products[27].count == 0
        assert len(zone3_session.cross_zone_returns) == 2

    def test_cross_zone_return_with_tolerance(self, store):
        """tolerance 범위 내에서 크로스 존 반환 매칭."""
        # Zone 1: 치킨마요 1개 (365g)
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 363g 반환 (365g와 2g 차이 - tolerance 5g 이내)
        trigger2 = self._create_return_trigger(zone=2, delta=363.0)
        store.add_trigger(zone=2, result=trigger2)

        # 확인: 매칭 성공
        zone1_session = store.get_session(zone=1)
        zone2_session = store.get_session(zone=2)

        assert zone1_session.aggregated_products[26].count == 0
        assert len(zone2_session.cross_zone_returns) == 1

    def test_cross_zone_return_outside_tolerance(self, store):
        """tolerance 범위 밖은 매칭 실패."""
        # Zone 1: 치킨마요 1개 (365g)
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 358g 반환 (365g와 7g 차이 - tolerance 5g 초과)
        trigger2 = self._create_return_trigger(zone=2, delta=358.0)
        store.add_trigger(zone=2, result=trigger2)

        # 확인: 매칭 실패 - unmatched로 유지
        zone1_session = store.get_session(zone=1)
        zone2_session = store.get_session(zone=2)

        assert zone1_session.aggregated_products[26].count == 1  # 그대로
        assert len(zone2_session.unmatched_returns) == 1
        assert len(zone2_session.cross_zone_returns) == 0

    def test_cross_zone_closest_match(self, store):
        """여러 상품이 tolerance 범위에 있으면 가장 가까운 것 선택."""
        # Zone 1: 치킨마요 1개 (365g)
        store.add_trigger(
            zone=1,
            result=self._create_removal_trigger(
                zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
            ),
        )

        # Zone 2: 28번 상품 1개 (400g)
        store.add_trigger(
            zone=2,
            result=self._create_removal_trigger(
                zone=2, delta=-400.0, product_id=28, name="김치볶음밥", count=1
            ),
        )

        # Zone 3: 398g 반환 - 치킨마요(365g, 차이 33g), 김치볶음밥(400g, 차이 2g)
        # tolerance=5g이면 김치볶음밥만 매칭 가능 (더 가까움)
        store.add_trigger(zone=3, result=self._create_return_trigger(zone=3, delta=398.0))

        # 확인: 김치볶음밥(400g)이 매칭됨
        zone2_session = store.get_session(zone=2)
        zone3_session = store.get_session(zone=3)

        assert zone2_session.aggregated_products[28].count == 0  # 차감됨
        assert len(zone3_session.cross_zone_returns) == 1
        assert zone3_session.cross_zone_returns[0].product_name == "김치볶음밥"

    def test_cross_zone_return_serialization(self, store):
        """CrossZoneReturn 직렬화/역직렬화 테스트."""
        # Zone 1: 치킨마요 2개
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-730.0, product_id=26, name="치킨마요", count=2
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 반환
        trigger2 = self._create_return_trigger(zone=2, delta=365.0)
        store.add_trigger(zone=2, result=trigger2)

        # to_dict / from_dict 테스트
        zone2_session = store.get_session(zone=2)
        session_dict = zone2_session.to_dict()

        # cross_zone_returns가 직렬화되어 있는지 확인
        assert "cross_zone_returns" in session_dict
        assert len(session_dict["cross_zone_returns"]) == 1
        assert session_dict["cross_zone_returns"][0]["source_zone"] == 2
        assert session_dict["cross_zone_returns"][0]["target_zone"] == 1

        # from_dict로 복원
        from model_service.session.door_session import DoorSession

        restored = DoorSession.from_dict(session_dict)
        assert len(restored.cross_zone_returns) == 1
        assert restored.cross_zone_returns[0].product_name == "치킨마요"


class TestCrossZoneReturnEdgeCases:
    """크로스 존 반환 엣지 케이스 테스트."""

    @pytest.fixture
    def store(self):
        """테스트용 DoorSessionStore 생성."""

        def get_weight(product_id: int) -> float:
            weights = {26: 365.0, 27: 250.0}
            return weights.get(product_id, 0.0)

        temp_dir = tempfile.mkdtemp()
        store = DoorSessionStore(
            yaml_dir=temp_dir,
            session_timeout=30.0,
            weight_tolerance=5.0,
            get_product_weight=get_weight,
        )
        yield store
        store.clear_all()

    def _create_removal_trigger(
        self, zone: int, delta: float, product_id: int, name: str, count: int
    ) -> TriggerResult:
        return TriggerResult(
            trigger_id="",
            session_id=f"zone_{zone}_{int(time.time())}",
            timestamp=time.time(),
            products=[
                ProductResult(
                    product_id=product_id,
                    product_idx=str(product_id),
                    name=name,
                    count=count,
                    price=3500,
                    confidence=0.95,
                )
            ],
            delta_weight=delta,
            confidence=0.95,
            video_paths={},
            is_return=False,
        )

    def _create_return_trigger(self, zone: int, delta: float) -> TriggerResult:
        return TriggerResult(
            trigger_id="",
            session_id=f"zone_{zone}_{int(time.time())}",
            timestamp=time.time(),
            products=[],
            delta_weight=delta,
            confidence=0.0,
            video_paths={},
            is_return=True,
        )

    def test_no_active_sessions_for_cross_zone(self, store):
        """다른 활성 zone이 없으면 unmatched 유지."""
        # Zone 1만 반환 trigger (다른 zone 없음)
        trigger = self._create_return_trigger(zone=1, delta=365.0)
        store.add_trigger(zone=1, result=trigger)

        zone1_session = store.get_session(zone=1)
        assert len(zone1_session.unmatched_returns) == 1
        assert len(zone1_session.cross_zone_returns) == 0

    def test_cross_zone_does_not_subtract_below_zero(self, store):
        """count가 0 이하인 상품은 차감하지 않음."""
        # Zone 1: 치킨마요 1개
        trigger1 = self._create_removal_trigger(
            zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
        )
        store.add_trigger(zone=1, result=trigger1)

        # Zone 2: 첫 번째 반환 -> Zone 1 count 0으로
        trigger2 = self._create_return_trigger(zone=2, delta=365.0)
        store.add_trigger(zone=2, result=trigger2)

        # Zone 3: 두 번째 반환 -> Zone 1 count가 이미 0이므로 매칭 안됨
        trigger3 = self._create_return_trigger(zone=3, delta=365.0)
        store.add_trigger(zone=3, result=trigger3)

        # 확인
        zone1_session = store.get_session(zone=1)
        zone3_session = store.get_session(zone=3)

        assert zone1_session.aggregated_products[26].count == 0  # 0 유지 (음수 안됨)
        assert len(zone3_session.unmatched_returns) == 1  # 매칭 실패
        assert len(zone3_session.cross_zone_returns) == 0

    def test_cross_zone_priority_order(self, store):
        """여러 zone에 같은 상품이 있으면 먼저 발견된 zone에서 차감."""
        # Zone 1: 치킨마요 1개
        store.add_trigger(
            zone=1,
            result=self._create_removal_trigger(
                zone=1, delta=-365.0, product_id=26, name="치킨마요", count=1
            ),
        )

        # Zone 2: 치킨마요 1개
        store.add_trigger(
            zone=2,
            result=self._create_removal_trigger(
                zone=2, delta=-365.0, product_id=26, name="치킨마요", count=1
            ),
        )

        # Zone 3: 반환 -> 먼저 순회되는 zone에서 차감
        store.add_trigger(zone=3, result=self._create_return_trigger(zone=3, delta=365.0))

        zone1_session = store.get_session(zone=1)
        zone2_session = store.get_session(zone=2)
        zone3_session = store.get_session(zone=3)

        # 둘 중 하나는 0, 하나는 1
        total_count = (
            zone1_session.aggregated_products[26].count
            + zone2_session.aggregated_products[26].count
        )
        assert total_count == 1
        assert len(zone3_session.cross_zone_returns) == 1
