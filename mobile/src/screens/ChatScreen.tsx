import React, {useState, useRef, useCallback, useEffect} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import Animated, {FadeInUp, FadeIn} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import {useAuth} from '../context/AuthContext';
import {useTradingState} from '../context/TradingStateContext';
import {PulsingDot} from '../components/PulsingDot';
import {EmptyState} from '../components/EmptyState';
import {colors, typography, spacing, borderRadius} from '../theme';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  action?: string;
}

const SUGGESTED_PROMPTS = [
  'How are positions doing?',
  'Should I take profits?',
  'Market conditions today?',
  'Analyze my recent trades',
];

function TypingIndicator() {
  return (
    <Animated.View entering={FadeIn.duration(200)} style={styles.typingContainer}>
      <View style={styles.typingDots}>
        {[0, 1, 2].map(i => (
          <Animated.View
            key={i}
            entering={FadeIn.delay(i * 150).duration(200)}
            style={styles.typingDot}
          />
        ))}
      </View>
    </Animated.View>
  );
}

export function ChatScreen() {
  const {apiClient} = useAuth();
  const {state} = useTradingState();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  const llmAvailable = state.llm?.available ?? false;

  const sendMessage = useCallback(async (text?: string) => {
    const msgText = (text || input).trim();
    if (!msgText || sending) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      text: msgText,
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setSending(true);

    try {
      const res = await apiClient.chatLLM(msgText);
      const botMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        text: res.response,
        action: res.action,
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err: any) {
      const errMsg: ChatMessage = {
        id: `e-${Date.now()}`,
        role: 'assistant',
        text: `Error: ${err.message}`,
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setSending(false);
    }
  }, [input, sending, apiClient]);

  const renderMessage = useCallback(({item, index}: {item: ChatMessage; index: number}) => {
    const isUser = item.role === 'user';
    return (
      <Animated.View entering={FadeInUp.delay(50).duration(250)}>
        <View style={[styles.bubble, isUser ? styles.userBubble : styles.botBubble]}>
          {!isUser && (
            <View style={styles.botHeader}>
              <Ionicons name="hardware-chip" size={14} color={colors.cyan} />
              <Text style={styles.botName}>Rudra</Text>
            </View>
          )}
          <Text style={[styles.bubbleText, isUser ? styles.userText : styles.botText]}>
            {item.text}
          </Text>
          {item.action && (
            <View style={styles.actionBadge}>
              <Ionicons name="flash" size={12} color={colors.yellow} />
              <Text style={styles.actionText}>{item.action}</Text>
            </View>
          )}
        </View>
      </Animated.View>
    );
  }, []);

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={90}
    >
      {/* LLM Status */}
      <View style={styles.statusBar}>
        <PulsingDot color={llmAvailable ? colors.green : colors.red} size={6} active={llmAvailable} />
        <Text style={[styles.statusText, {color: llmAvailable ? colors.green : colors.red}]}>
          {llmAvailable ? 'LLM Connected' : 'LLM Offline'}
        </Text>
        {state.llm?.model && (
          <Text style={styles.modelText}>{state.llm.model}</Text>
        )}
      </View>

      {messages.length === 0 ? (
        <View style={styles.emptyContainer}>
          <EmptyState
            icon="chatbubbles-outline"
            title="Rudra Supervisor"
            subtitle="Ask about positions, market conditions, or request trading actions"
          />
          <View style={styles.promptChips}>
            {SUGGESTED_PROMPTS.map(prompt => (
              <TouchableOpacity
                key={prompt}
                style={styles.promptChip}
                onPress={() => sendMessage(prompt)}
                activeOpacity={0.7}
              >
                <Text style={styles.promptChipText}>{prompt}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      ) : (
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.messageList}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
          ListFooterComponent={sending ? <TypingIndicator /> : null}
        />
      )}

      {/* Input bar */}
      <View style={styles.inputBar}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask Rudra..."
          placeholderTextColor={colors.textMuted}
          multiline
          maxLength={2000}
          editable={!sending}
          onSubmitEditing={() => sendMessage()}
          blurOnSubmit={false}
        />
        <TouchableOpacity
          style={[styles.sendBtn, (!input.trim() || sending) && styles.sendBtnDisabled]}
          onPress={() => sendMessage()}
          disabled={!input.trim() || sending}
          activeOpacity={0.7}
        >
          <Ionicons name="send" size={18} color={colors.bg} />
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  statusBar: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    paddingHorizontal: spacing.lg, paddingVertical: spacing.sm,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  statusText: {...typography.caption, fontWeight: '600'},
  modelText: {...typography.caption, color: colors.textMuted, marginLeft: 'auto'},
  emptyContainer: {flex: 1, justifyContent: 'center'},
  promptChips: {flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: spacing.sm, paddingHorizontal: spacing.xl},
  promptChip: {
    backgroundColor: colors.bgCard, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.round, paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
  },
  promptChipText: {...typography.caption, color: colors.cyan},
  messageList: {padding: spacing.md, paddingBottom: spacing.sm},
  bubble: {maxWidth: '80%', padding: spacing.md, borderRadius: borderRadius.xl, marginBottom: spacing.sm},
  userBubble: {backgroundColor: colors.cyan, alignSelf: 'flex-end', borderBottomRightRadius: 4},
  botBubble: {backgroundColor: colors.bgCard, alignSelf: 'flex-start', borderBottomLeftRadius: 4, borderWidth: 1, borderColor: colors.border},
  botHeader: {flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: spacing.xs},
  botName: {...typography.label, color: colors.cyan, fontSize: 10},
  bubbleText: {...typography.body, lineHeight: 20},
  userText: {color: colors.bg},
  botText: {color: colors.textPrimary},
  actionBadge: {flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginTop: spacing.sm, backgroundColor: 'rgba(245, 158, 11, 0.1)', paddingHorizontal: spacing.sm, paddingVertical: 3, borderRadius: borderRadius.sm},
  actionText: {...typography.caption, color: colors.yellow},
  typingContainer: {alignSelf: 'flex-start', backgroundColor: colors.bgCard, borderRadius: borderRadius.xl, borderBottomLeftRadius: 4, padding: spacing.md, borderWidth: 1, borderColor: colors.border},
  typingDots: {flexDirection: 'row', gap: 4},
  typingDot: {width: 6, height: 6, borderRadius: 3, backgroundColor: colors.textMuted},
  inputBar: {
    flexDirection: 'row', padding: spacing.md, borderTopWidth: 1, borderTopColor: colors.border,
    backgroundColor: colors.bgCard, alignItems: 'flex-end', gap: spacing.sm,
  },
  input: {
    flex: 1, backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.xl, paddingHorizontal: spacing.lg, paddingVertical: spacing.sm,
    color: colors.textPrimary, ...typography.body, maxHeight: 100,
  },
  sendBtn: {
    backgroundColor: colors.cyan, width: 40, height: 40,
    borderRadius: 20, justifyContent: 'center', alignItems: 'center',
  },
  sendBtnDisabled: {opacity: 0.4},
});
