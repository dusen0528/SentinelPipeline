/**
 * EventMapper - Application Layer
 * 백엔드의 위험 키워드 분석 결과를 프론트엔드 이벤트로 변환
 */
import { Event } from '../domain/Event.js';

export class EventMapper {
    /**
     * 위험 유형을 이벤트 타입으로 매핑
     */
    static mapRiskTypeToEventType(riskType) {
        const mapping = {
            'FIRE': 'FIRE_DETECTED',
            'HELP': 'KEYWORD_DETECTED',
            'THREAT': 'KEYWORD_DETECTED',
            'EMERGENCY': 'KEYWORD_DETECTED',
            'NONE': null
        };
        return mapping[riskType] || 'KEYWORD_DETECTED';
    }

    /**
     * 백엔드 위험 분석 결과를 프론트엔드 이벤트로 변환
     */
    static fromRiskAnalysis(riskResult, streamId, sttText) {
        if (!riskResult.is_dangerous) {
            return null;
        }

        const eventType = this.mapRiskTypeToEventType(riskResult.event_type);
        
        return new Event(
            eventType,
            streamId,
            {
                keyword: riskResult.keyword,
                text: riskResult.original_text || sttText,
                confidence: riskResult.confidence,
                risk_type: riskResult.event_type
            }
        );
    }

    /**
     * 백엔드 이벤트 페이로드를 프론트엔드 이벤트로 변환
     */
    static fromBackendPayload(payload) {
        return Event.fromJSON({
            event_type: payload.event_type,
            stream_id: payload.stream_id,
            data: payload.data,
            timestamp: payload.timestamp
        });
    }
}

