"""
Service Container module for dependency injection.

의존성 주입을 위한 서비스 컨테이너 모듈.
전역 싱글톤 대신 명시적인 컨테이너 인스턴스로 관리합니다.

v4.2 추가:
- ServiceContainer: 의존성 라이프사이클 관리
- 테스트 격리 지원
"""

from .service_container import ServiceContainer

__all__ = [
    "ServiceContainer",
]
